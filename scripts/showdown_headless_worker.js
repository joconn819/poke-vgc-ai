#!/usr/bin/env node
"use strict";

// Persistent line-oriented adapter around the local Showdown Battle API.
// Standard output is reserved for JSON protocol messages; diagnostics go to
// stderr so Python policy rollouts can safely stream choices and requests.
const path = require("path");
const readline = require("readline");

const root = process.env.POKEMON_SHOWDOWN_PATH;
if (!root) throw new Error("Set POKEMON_SHOWDOWN_PATH to the local Showdown checkout");
const { Battle, Dex } = require(path.join(root, "dist/sim"));
const { PRNG } = require(path.join(root, "dist/sim/prng"));
const { State } = require(path.join(root, "dist/sim/state"));
const { generateTeam } = require("./showdown_random_team");
const { projectLogForPlayer } = require("./showdown_player_log");
const {
  alignHypothesisRosterToLive,
  buildPublicCompatibleSnapshot,
} = require("./showdown_hidden_snapshot");

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
const input = rl[Symbol.asyncIterator]();
let battle;
let logOffset = 0;

async function nextInput() {
  const value = await input.next();
  if (value.done) throw new Error("parent closed the rollout protocol before match completion");
  return JSON.parse(value.value);
}

function protocolId(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function publicBattleIdentityByNickname(log, sidePrefix = "p2") {
  const rosterSpecies = new Map();
  const activeSpecies = new Map();
  const transformed = new Set();
  for (const rawLine of log) {
    const parts = String(rawLine).split("|");
    if (parts.length < 3) continue;
    const actor = parts[2].trim();
    if (!actor.toLowerCase().startsWith(sidePrefix)) continue;
    const colon = actor.indexOf(":");
    if (colon < 0) continue;
    const nickname = protocolId(actor.slice(colon + 1));
    if (!nickname) continue;
    if (["switch", "drag", "replace"].includes(parts[1]) && parts.length >= 4) {
      const rawSpecies = parts[3].split(",", 1)[0].trim();
      const visibleSpecies = Dex.species.get(rawSpecies);
      const rosterName = visibleSpecies.battleOnly || visibleSpecies.name;
      if (rosterName) {
        rosterSpecies.set(nickname, rosterName);
        activeSpecies.set(nickname, visibleSpecies.name || rosterName);
        transformed.delete(nickname);
      }
    } else if (["-mega", "-formechange", "detailschange"].includes(parts[1]) && parts.length >= 4) {
      activeSpecies.set(nickname, Dex.species.get(parts[3]).name);
    } else if (parts[1] === "-transform") {
      transformed.add(nickname);
    }
  }
  return {rosterSpecies, activeSpecies, transformed};
}

function publishState() {
  const requests = battle.getRequests(battle.requestState);
  const logDelta = battle.log.slice(logOffset);
  const payload = {
    type: "state",
    format: battle.format.id,
    request_state: battle.requestState,
    turn: battle.turn,
    ended: battle.ended,
    winner: battle.winner || null,
    requests,
    // `Battle.log` retains both branches of |split| events for simulator and
    // replay use. Never send that omniscient buffer to a player view.
    log: projectLogForPlayer(logDelta, null),
    player_logs: [
      projectLogForPlayer(logDelta, "p1"),
      projectLogForPlayer(logDelta, "p2"),
    ],
  };
  logOffset = battle.log.length;
  process.stdout.write(JSON.stringify(payload) + "\n");
}

function swapSerializedSideReferences(value) {
  if (typeof value === "string") {
    const marker = "\u0000SHOWDOWN_SIDE_SWAP\u0000";
    return value
      .replace(/p1(?=(?:[ab](?::|\])|[^a-z0-9]|$))/gi, marker)
      .replace(/p2(?=(?:[ab](?::|\])|[^a-z0-9]|$))/gi, "p1")
      .replaceAll(marker, "p2");
  }
  if (Array.isArray(value)) return value.map(swapSerializedSideReferences);
  if (value && typeof value === "object") {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = swapSerializedSideReferences(item);
    }
    return result;
  }
  return value;
}

function snapshotFromActingSide(actingSide) {
  const snapshot = State.serializeBattle(battle);
  if (actingSide === 0) return snapshot;
  const swapped = swapSerializedSideReferences(snapshot);
  swapped.sides = [swapped.sides[1], swapped.sides[0]];
  return swapped;
}

function buildSanitizedSearchWorlds(message, ownTeam) {
  if (!battle || battle.ended || battle.requestState !== "move") {
    throw new Error("midgame search worlds require a live move request");
  }
  const actingSide = Number(message.acting_side ?? 0);
  if (![0, 1].includes(actingSide)) {
    throw new Error("acting_side must be 0 (P1) or 1 (P2)");
  }
  const ownPreview = message.own_preview;
  const opponentSpecies = message.opponent_preview_species;
  const hypotheses = message.hypotheses;
  const factsBySpecies = message.revealed_facts || {};
  if (!Array.isArray(ownPreview) || ownPreview.length !== 4 || new Set(ownPreview).size !== 4) {
    throw new Error("own_preview must contain four distinct team indices");
  }
  if (!Array.isArray(opponentSpecies) || opponentSpecies.length !== 6) {
    throw new Error("opponent_preview_species must contain six public species");
  }
  if (!Array.isArray(hypotheses) || hypotheses.length < 1) {
    throw new Error("at least one hidden-team hypothesis is required");
  }

  const liveSnapshot = snapshotFromActingSide(actingSide);
  // The matrix runner assumes the acting player's perspective is P1. Compare
  // its active-slot order to the actual live side before constructing action
  // candidates; active Pokemon may have changed since team preview.
  const expectedOwnActives = battle.sides[actingSide].active.map(pokemon =>
    protocolId(pokemon?.set?.species || pokemon?.species)
  );
  const normalizedOwnActives = State.deserializeBattle(liveSnapshot).sides[0].active.map(
    pokemon => protocolId(pokemon?.set?.species || pokemon?.species)
  );
  if (
    expectedOwnActives.length !== 2
    || normalizedOwnActives.length !== 2
    || expectedOwnActives.some((species, index) => species !== normalizedOwnActives[index])
  ) {
    throw new Error(
      `acting-side snapshot mismatch: live own actives ${expectedOwnActives.join(",")}; `
      + `normalized P1 actives ${normalizedOwnActives.join(",")}`
    );
  }
  const opponentSide = battle.sides[1 - actingSide];
  const previewSpecies = opponentSpecies.map(name => Dex.species.get(name));
  // An unrevealed Illusion can make one public switch name correspond to a
  // different selected roster member. Until that hidden identity is modeled as
  // a belief, disable search for every preview that could contain Illusion.
  // This gate depends only on public species, never hidden simulator fields.
  const illusionInPreview = previewSpecies.some(species =>
    Object.values(species.abilities || {}).some(ability => protocolId(ability) === "illusion")
  );
  if (illusionInPreview) {
    throw new Error("midgame search skips public previews containing a possible Illusion user");
  }
  const publicIdentityByNickname = publicBattleIdentityByNickname(liveSnapshot.log);
  const publicSpeciesByPosition = {};
  const publicStateByPosition = {};
  const publicFormChanges = new Map();
  for (const pokemon of opponentSide.pokemon) {
    if (!(pokemon.isActive || pokemon.activeTurns > 0 || pokemon.fainted)) continue;
    const position = Number(pokemon.position);
    const nickname = protocolId(pokemon.name);
    if (publicIdentityByNickname.transformed.has(nickname)) {
      throw new Error("midgame search skips publicly observed Transform states");
    }
    const rosterName = publicIdentityByNickname.rosterSpecies.get(nickname);
    if (!rosterName) {
      throw new Error("no public switch identity is available for a revealed opponent Pokemon");
    }
    const activeName = publicIdentityByNickname.activeSpecies.get(nickname) || rosterName;
    if (protocolId(activeName) !== protocolId(rosterName)) {
      publicFormChanges.set(rosterName, activeName);
    }
    const rosterSpecies = Dex.species.get(rosterName);
    const exactMatches = previewSpecies.filter(candidate => candidate.id === rosterSpecies.id);
    const matched = exactMatches.length === 1 ? exactMatches[0] : null;
    if (!matched) {
      throw new Error(`could not map public switch identity ${rosterName} to one preview species`);
    }
    const species = matched.name;
    const speciesId = matched.id;
    const facts = factsBySpecies[speciesId] || {};
    const sharedCondition = pokemon.fainted ? "0/100 fnt" : pokemon.getHealth().shared;
    const hpMatch = /^(\d+)\/100/.exec(sharedCondition);
    if (!hpMatch) throw new Error(`unexpected public HP condition for ${species}: ${sharedCondition}`);
    publicSpeciesByPosition[position] = species;
    publicStateByPosition[position] = {
      publicSpecies: species,
      publicHpFraction: Number.isFinite(facts.hp_fraction)
        ? Number(facts.hp_fraction)
        : Number(hpMatch[1]) / 100,
      revealedMoveIds: Array.isArray(facts.moves) ? facts.moves : [],
      ...(facts.item ? {revealedItem: facts.item} : {}),
      ...(facts.ability ? {revealedAbility: facts.ability} : {}),
      ...(facts.active_ability ? {activeAbility: facts.active_ability} : {}),
      ...(facts.tera_type ? {
        revealedTeraType: facts.tera_type,
        teraTypeRevealed: true,
      } : {}),
    };
  }

  const snapshots = [];
  const worldMetadata = [];
  for (let worldIndex = 0; worldIndex < hypotheses.length; worldIndex++) {
    const hypothesis = hypotheses[worldIndex];
    if (!Array.isArray(hypothesis.opponent_team) || hypothesis.opponent_team.length !== 6) {
      throw new Error(`hidden world ${worldIndex} must contain six sampled opponent sets`);
    }
    if (!Array.isArray(hypothesis.opponent_preview) || hypothesis.opponent_preview.length !== 4) {
      throw new Error(`hidden world ${worldIndex} must contain a four-Pokemon preview`);
    }
    const template = new Battle({
      formatid: battle.format.id,
      seed: [message.seed + worldIndex * 1009, 17, 29, 43],
      strictChoices: true,
      p1: {name: "Search P1", team: structuredClone(ownTeam)},
      p2: {name: "Search P2 hypothesis", team: structuredClone(hypothesis.opponent_team)},
    });
    try {
      const ownCommand = `team ${ownPreview.map(index => Number(index) + 1).join("")}`;
      const opponentCommand = `team ${hypothesis.opponent_preview.map(index => Number(index) + 1).join("")}`;
      template.makeChoices(ownCommand, opponentCommand);
      if (template.ended || template.requestState !== "move") {
        throw new Error(`hidden world ${worldIndex} did not reach a move request`);
      }
    for (const [rosterName, activeName] of publicFormChanges) {
        const rosterId = Dex.species.get(rosterName).id;
        const matches = template.sides[1].pokemon.filter(pokemon =>
          Dex.species.get(pokemon.set.species).id === rosterId
        );
        if (matches.length !== 1) {
          throw new Error(`public forme identity ${rosterName} does not match one sampled set`);
        }
        if (!matches[0].formeChange(activeName, null, true)) {
          throw new Error(`could not apply publicly observed forme ${activeName} in hidden world`);
        }
      }
      const aligned = alignHypothesisRosterToLive(
        liveSnapshot,
        State.serializeBattle(template),
        1,
        {publicSpeciesByPosition},
      );
      snapshots.push(buildPublicCompatibleSnapshot(
        aligned.liveSnapshot,
        aligned.hypothesisSnapshot,
        1,
        {publicPreviewSpecies: opponentSpecies, publicStateByPosition},
      ));
      worldMetadata.push({
        world_index: worldIndex,
        selected_roster: hypothesis.opponent_preview,
        own_team: structuredClone(ownTeam),
        opponent_team: hypothesis.opponent_team.map(pokemon => ({
          species: pokemon.species || pokemon.speciesId || pokemon.name,
        })),
      });
    } finally {
      template.destroy();
    }
  }
  return {snapshots, worlds: worldMetadata};
}

async function playMatch(init) {
  logOffset = 0;
  const formatId = init.format || "gen9championsdoubles2026mcclosed";
  const format = Dex.formats.get(formatId);
  if (!format.exists || format.gameType !== "doubles") throw new Error(`unsupported format ${formatId}`);
  if (!Array.isArray(init.team1) || init.team1.length !== 6 || !Array.isArray(init.team2) || init.team2.length !== 6) {
    throw new Error("headless match requires exactly six Pokemon per player");
  }
  const ruleTable = Dex.formats.getRuleTable(format);
  if (ruleTable.has("openteamsheets")) throw new Error("headless closed-sheet worker refuses open-team-sheet formats");
  battle = init.snapshot ? Battle.fromJSON(init.snapshot) : new Battle({
    formatid: formatId,
    seed: [init.seed || 1, (init.seed || 1) + 1, (init.seed || 1) + 2, (init.seed || 1) + 3],
    strictChoices: true,
    p1: {name: "Headless P1", team: init.team1},
    p2: {name: "Headless P2", team: init.team2},
  });
  // Showdown's generic State serializer omits the battle-level requestState.
  // Counterfactual branches need to resume the exact active request so strict
  // choice handling and search-world preparation remain valid.
  if (init.snapshot?.requestState) {
    battle.requestState = init.snapshot.requestState;
  }
  if (init.snapshot && (battle.ended || !battle.requestState)) {
    throw new Error("restored counterfactual snapshot must have a live request");
  }
  if (init.snapshot) logOffset = battle.log.length;
  publishState();
  while (!battle.ended) {
    const message = await nextInput();
    if (message.type === "snapshot") {
      const snapshot = State.serializeBattle(battle);
      snapshot.requestState = battle.requestState;
      process.stdout.write(JSON.stringify({type: "snapshot", snapshot}) + "\n");
      continue;
    }
    if (message.type === "search_worlds") {
      try {
        const actingSide = Number(message.acting_side ?? 0);
        const ownTeam = actingSide === 0 ? init.team1 : init.team2;
        const result = buildSanitizedSearchWorlds(message, ownTeam);
        process.stdout.write(JSON.stringify({type: "search_worlds", ...result}) + "\n");
      } catch (error) {
        process.stdout.write(JSON.stringify({
          type: "search_worlds",
          error: String(error),
          stack: error.stack || "",
        }) + "\n");
      }
      continue;
    }
    if (message.type === "reseed_random") {
      if (battle.ended || !Number.isSafeInteger(message.seed)) {
        throw new Error("reseed_random requires a live battle and safe integer seed");
      }
      const seedHex = BigInt.asUintN(64, BigInt(message.seed)).toString(16).padStart(16, "0");
      battle.prng.setSeed(`gen5,${seedHex}`);
      process.stdout.write(JSON.stringify({type: "random_seed_set", seed: message.seed}) + "\n");
      continue;
    }
    if (message.type === "abort_match") {
      process.stdout.write(JSON.stringify({type: "match_aborted"}) + "\n");
      break;
    }
    if (message.type !== "choices" || !Array.isArray(message.choices) || message.choices.length !== 2) {
      throw new Error("expected {type:'choices', choices:[p1, p2]}");
    }
    if (process.env.POKE_HEADLESS_DEBUG) {
      const activeState = side => ({
        active: side.active.map(pokemon => pokemon?.name || null),
        requestActive: (side.activeRequest?.active || []).map(request => ({
          trapped: Boolean(request.trapped),
          maybeTrapped: Boolean(request.maybeTrapped),
          forceSwitch: Boolean(request.forceSwitch),
        })),
      });
      process.stderr.write(`before makeChoices turn=${battle.turn} state=${battle.requestState} choices=${JSON.stringify(message.choices)} p1=${JSON.stringify(activeState(battle.p1))} p2=${JSON.stringify(activeState(battle.p2))}\n`);
    }
    battle.makeChoices(message.choices[0], message.choices[1]);
    if (process.env.POKE_HEADLESS_DEBUG) {
      process.stderr.write(`after makeChoices turn=${battle.turn} ended=${battle.ended}\n`);
    }
    publishState();
  }
  battle.destroy();
}

async function main() {
  while (true) {
    const init = await nextInput();
    if (init.type === "shutdown") break;
    if (init.type === "generate_team") {
      const team = generateTeam(init.seed, init.format || "gen9championsdoubles2026mcclosed", init.archetype || "random");
      process.stdout.write(JSON.stringify({type: "team", team}) + "\n");
      continue;
    }
    if (init.type !== "start_match") throw new Error(`unexpected command ${init.type}`);
    await playMatch(init);
  }
  rl.close();
}

main().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  rl.close();
  process.exitCode = 1;
});
