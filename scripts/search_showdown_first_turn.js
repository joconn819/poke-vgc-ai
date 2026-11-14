#!/usr/bin/env node
"use strict";

// Simultaneous one-turn search from the local Showdown engine. The caller
// supplies player-visible team hypotheses or explicitly sanitized current-state
// snapshots; the engine's actual hidden team is never accepted as a search world.
const path = require("path");
const os = require("os");
const readline = require("readline");
const root = process.env.POKEMON_SHOWDOWN_PATH;
if (!root) throw new Error("Set POKEMON_SHOWDOWN_PATH to the local Showdown checkout");
const { Battle, Dex } = require(path.join(root, "dist/sim"));
const { State } = require(path.join(root, "dist/sim/state"));
const { PRNG } = require(path.join(root, "dist/sim/prng"));

const formatId = process.argv[2] || "gen9championsdoubles2026mcclosed";
function runSearch(args) {
    const searchFormatId = args.format || formatId;
    const maxCandidates = Math.max(2, Number(args.max_candidates || 10));
    const maxResponseCandidates = Math.max(
      1,
      Number(args.max_response_candidates || maxCandidates),
    );
    const maxDamageResponseCandidates = Math.max(
      1,
      Math.floor(Number(args.max_damage_response_candidates || 1)),
    );
    const responseSelectionStrategy = args.response_selection_strategy || "semantic";
    if (!["semantic", "slot_coverage", "damage_ranked"].includes(responseSelectionStrategy)) {
      throw new Error(`unsupported response selection strategy: ${responseSelectionStrategy}`);
    }
    const chanceSamples = Math.max(1, Math.floor(Number(args.chance_samples || 1)));
    const candidateDamageSamples = Math.max(
      1, Math.floor(Number(args.candidate_damage_samples || 4)),
    );
    const maxChanceSamples = Math.max(
      chanceSamples,
      Math.floor(Number(args.max_chance_samples || chanceSamples)),
    );
    const confidenceMultiplier = Math.max(0, Number(args.confidence_multiplier || 2));
    const seed = Number(args.seed || 1);
    const worldStartIndex = Number(args.world_start_index || 0);
    const returnFrontiers = Boolean(args.return_frontiers);
    const returnProtocolMessages = Boolean(args.return_protocol_messages);
    const returnResponseCandidates = Boolean(args.return_response_candidates);
    const preferredOwnAction = args.preferred_own_action;
    const initialSnapshots = args.sanitized_snapshots;
    const suppliedWorlds = args.worlds;
    if (initialSnapshots !== undefined && (!Array.isArray(initialSnapshots) || initialSnapshots.length < 1)) {
      throw new Error("sanitized_snapshots must be a non-empty array when provided");
    }
    if (initialSnapshots === undefined && (!Array.isArray(suppliedWorlds) || suppliedWorlds.length < 1)) {
      throw new Error("worlds must be a non-empty array when sanitized_snapshots are absent");
    }
    if (initialSnapshots !== undefined && suppliedWorlds !== undefined
      && (!Array.isArray(suppliedWorlds) || suppliedWorlds.length !== initialSnapshots.length)) {
      throw new Error("world metadata count must match sanitized_snapshots");
    }
    if (returnFrontiers && initialSnapshots !== undefined && suppliedWorlds === undefined) {
      throw new Error("world metadata is required when returning frontiers from sanitized snapshots");
    }
    const worlds = suppliedWorlds || initialSnapshots.map(() => ({}));
    if (worlds.length < 1) throw new Error("search needs at least one world");
    const format = Dex.formats.get(searchFormatId);
    if (!format.exists || format.gameType !== "doubles") throw new Error(`Unsupported format ${searchFormatId}`);

    function requestForSide(battle, sideIndex) {
      const side = battle.sides[sideIndex];
      if (side && side.activeRequest !== undefined) return side.activeRequest;
      return battle.getRequests(battle.requestState)[sideIndex];
    }

    function targetLocations(target, sideIndex, activeIndex, battle) {
      const pokemon = battle.sides[sideIndex].active[activeIndex];
      let targets;
      // randomNormal moves (for example Outrage) choose their target in the
      // engine. Supplying an explicit target makes Showdown reject the order.
      if (target === "randomNormal") targets = [null];
      else if (target === "normal") targets = [-2, -1, 1, 2];
      else if (target === "any") targets = [-2, -1, 1, 2];
      else if (target === "adjacentFoe") targets = [1, 2];
      else if (target === "adjacentAlly") targets = [-2, -1];
      else if (target === "adjacentAllyOrSelf") targets = [-2, -1, 0];
      else targets = [null];
      return targets.filter(targetLoc => {
        if (targetLoc == null) return true;
        if (targetLoc < 0 && !battle.sides[sideIndex].active[-targetLoc - 1]) return false;
        if (targetLoc > 0 && !battle.sides[1 - sideIndex].active[targetLoc - 1]) return false;
        return battle.validTargetLoc(targetLoc, pokemon, target);
      });
    }

    // Estimate damage with Showdown's own damage calculator so lower-power,
    // type-effective moves are not discarded. This estimate is a shortlist
    // heuristic only; turn RNG is restored after each calculation, and battle
    // rollouts remain the source of action payoffs.
    let estimatedDamageScoringNs = 0n;
    function estimatedDamageScore(battle, sideIndex, activeIndex, move, targetLoc) {
      const started = process.hrtime.bigint();
      try {
        return computeEstimatedDamageScore(battle, sideIndex, activeIndex, move, targetLoc);
      } finally {
        estimatedDamageScoringNs += process.hrtime.bigint() - started;
      }
    }

    function computeEstimatedDamageScore(battle, sideIndex, activeIndex, move, targetLoc) {
      if (move.category === "Status") return {shortlist: 0, raw: 0};
      const source = battle.sides[sideIndex].active[activeIndex];
      if (!source) return {shortlist: 0, raw: 0};
      const opponentSide = battle.sides[1 - sideIndex];
      const ownSide = battle.sides[sideIndex];
      let targets = [];
      if (targetLoc != null) {
        const targetSide = targetLoc > 0 ? opponentSide : ownSide;
        const targetIndex = targetLoc > 0 ? targetLoc - 1 : -targetLoc - 1;
        if (targetLoc === 0) targets = [source];
        else targets = [targetSide.active[targetIndex]].filter(Boolean);
      } else if (move.target === "allAdjacentFoes") {
        targets = opponentSide.active.filter(Boolean);
      } else if (move.target === "allAdjacent") {
        targets = [...opponentSide.active, ...ownSide.active].filter(Boolean);
      } else if (move.target === "all") {
        targets = [...opponentSide.active, ...ownSide.active].filter(Boolean);
      } else if (["normal", "any", "randomNormal", "adjacentFoe"].includes(move.target)) {
        targets = opponentSide.active.filter(Boolean);
      } else if (["adjacentAlly", "adjacentAllyOrSelf"].includes(move.target)) {
        targets = ownSide.active.filter(Boolean);
      } else if (move.target === "self") {
        targets = [source];
      }
      if (!targets.length) return {shortlist: 0, raw: 0};

      const expectedTargets = targets.map(target => {
        const isSpread = ["allAdjacentFoes", "allAdjacent", "all"].includes(move.target)
          && targets.length > 1;
        const hitChance = move.accuracy === true ? 1
          : Math.max(0, Math.min(1, Number(move.accuracy || 0) / 100));
        const savedPrng = battle.prng;
        let totalDamage = 0;
        const samples = candidateDamageSamples;
        try {
          for (let sample = 0; sample < samples; sample++) {
            battle.prng = new PRNG([sample + 1, sample + 11, sample + 101, sample + 1009]);
            const activeMove = battle.dex.getActiveMove(move.id);
            activeMove.willCrit = false;
            activeMove.spreadHit = isSpread;
            const damage = battle.actions.getDamage(source, target, activeMove, true);
            totalDamage += Number.isFinite(damage) ? Math.max(0, damage) : 0;
          }
        } catch (_error) {
          return {shortlist: 0, raw: 0};
        } finally {
          battle.prng = savedPrng;
        }
        const meanDamage = totalDamage / samples * hitChance;
        const maxHp = Math.max(1, Number(target.maxhp || 1));
        const remainingHp = Math.max(1, Number(target.hp || 1));
        const isOpponent = target.side === opponentSide;
        const direction = isOpponent ? 1 : -1;
        const cappedDamage = Math.min(remainingHp, meanDamage);
        const knockoutBonus = meanDamage >= remainingHp ? 0.25 : 0;
        return {
          shortlist: direction * (cappedDamage / maxHp + knockoutBonus),
          raw: direction * meanDamage,
        };
      });
      return {
        shortlist: expectedTargets.reduce((sum, score) => sum + score.shortlist, 0),
        raw: expectedTargets.reduce((sum, score) => sum + score.raw, 0),
      };
    }

    function nativeRosterIndex(battle, request, sideIndex, requestIndex) {
      const requestPokemon = request.side.pokemon[requestIndex];
      const role = sideIndex === 0 ? "p1" : "p2";
      const previewRoster = battle.log
        .filter(line => {
          const parts = String(line).split("|");
          return parts[1] === "poke" && parts[2] === role;
        })
        .map(line => Dex.species.get(String(line).split("|")[3].split(",")[0]).id);
      const species = Dex.species.get(
        String(requestPokemon.details || "").split(",")[0].trim(),
      ).id;
      const index = previewRoster.indexOf(species);
      return index >= 0 ? index : requestIndex;
    }

    function nativeMoveOrder(battle, request, sideIndex, activeIndex) {
      // Preserve the Pokemon's native move slots. A locked Showdown request
      // can contain only one move even when that move occupies slot 2, 3, or
      // 4 in the agent's action space. Poke-env maps orders through these
      // original slots, so using the shortened request array shifts indices.
      const active = battle.sides[sideIndex].active[activeIndex];
      const slots = (active?.moveSlots || []).map(slot => Dex.moves.get(slot.id).id);
      if (slots.length) return slots;
      return (request.active?.[activeIndex]?.moves || [])
        .map(slot => Dex.moves.get(slot.id).id);
    }

    function individualOptions(battle, request, sideIndex, activeIndex) {
      const options = [];
      const activeRequest = request.active?.[activeIndex];
      if (request.forceSwitch) {
        if (!request.forceSwitch[activeIndex]) return [{command: "pass", kinds: ["pass", "defensive"], score: 0, action_index: 0, action: {type: "pass"}}];
        for (let i = 0; i < request.side.pokemon.length; i++) {
          const pokemon = request.side.pokemon[i];
          if (!pokemon.active && !pokemon.condition.endsWith(" fnt")) {
            options.push({command: `switch ${i + 1}`, kinds: ["switch", "defensive"], score: 0.2, action_index: nativeRosterIndex(battle, request, sideIndex, i) + 1, action: {type: "switch", species: Dex.species.get(String(pokemon.details || "").split(",")[0].trim()).id}});
          }
        }
        return options;
      }
      // Showdown can leave request.forceSwitch unset while a fainted active
      // still needs replacement in a multi-slot move request. Its request
      // object may retain stale move slots; those cannot be selected legally.
      const activePokemon = battle.sides[sideIndex]?.active?.[activeIndex];
      if (
        activePokemon?.fainted
        || (!activePokemon && request.side?.pokemon?.[activeIndex]?.condition?.endsWith(" fnt"))
      ) {
        for (let i = 0; i < request.side.pokemon.length; i++) {
          const pokemon = request.side.pokemon[i];
          if (!pokemon.active && !pokemon.condition.endsWith(" fnt")) {
            options.push({command: `switch ${i + 1}`, kinds: ["switch", "defensive"], score: 0.2, action_index: nativeRosterIndex(battle, request, sideIndex, i) + 1, action: {type: "switch", species: Dex.species.get(String(pokemon.details || "").split(",")[0].trim()).id}});
          }
        }
        if (!options.length) {
          return [{command: "pass", kinds: ["pass", "defensive"], score: 0, action_index: 0, action: {type: "pass"}}];
        }
        return options;
      }
      if (!activeRequest) return [{command: "pass", kinds: ["pass", "defensive"], score: 0, action_index: 0, action: {type: "pass"}}];
      const requestMoves = activeRequest.moves || [];
      const nativeMoves = nativeMoveOrder(battle, request, sideIndex, activeIndex);
      for (let i = 0; i < nativeMoves.length; i++) {
        const moveId = nativeMoves[i];
        const slot = requestMoves.find(moveSlot => moveSlot.id === moveId);
        if (!slot) continue;
        if (slot.disabled || (slot.pp != null && Number(slot.pp) <= 0)) continue;
        const move = Dex.moves.get(slot.id);
        const requestMoveIndex = requestMoves.indexOf(slot);
        const activePokemon = battle.sides[sideIndex].active[activeIndex];
        const hardLockedMove = requestMoves.length === 1 && slot.pp == null;
        let targetLocs;
        let forcedTargetless = false;
        if (["fight", "struggle", "recharge"].includes(move.id)) {
          // These special requests do not accept an explicit target.
          targetLocs = [null];
        } else if (hardLockedMove) {
          // A locked charge move may still list target locations in the
          // request's `options`, but Showdown rejects an explicit target when
          // parsing the choice (e.g. Solar Beam on its charge turn). The
          // battle's own `side.choose` validation below is authoritative.
          forcedTargetless = true;
          targetLocs = [null];
        } else if (
          move.nonGhostTarget
          && !(activePokemon.types || []).includes("Ghost")
        ) {
          targetLocs = [null];
        } else if (move.id === "pollenpuff" && activePokemon.volatiles?.healblock) {
          targetLocs = targetLocations("adjacentFoe", sideIndex, activeIndex, battle);
        } else if (
          move.id === "terastarstorm"
          && activePokemon.terastallized
          && activePokemon.teraType === "Stellar"
        ) {
          targetLocs = [null];
        } else {
          // Some Showdown request versions omit `target` from individual
          // move slots even though the move's Dex data has it. Falling back
          // to no target would create a candidate the live request rejects.
          targetLocs = targetLocations(slot.target || move.target, sideIndex, activeIndex, battle);
        }
        const kinds = [];
        if (["protect", "detect", "wideguard", "quickguard", "kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark"].includes(move.id)) {
          kinds.push("protect", "defensive");
        }
        if (move.boosts || move.sideCondition || move.volatileStatus === "trickroom" || ["tailwind", "reflect", "lightscreen", "auroraveil"].includes(move.id)) {
          kinds.push("setup", "defensive");
        }
        if (["followme", "ragepowder", "allyswitch"].includes(move.id)) {
          kinds.push("redirection", "defensive");
        }
        if (["helpinghand", "wideguard", "quickguard"].includes(move.id)) {
          kinds.push("defensive");
        }
        if (move.basePower > 0 || move.damage || move.damageCallback) kinds.push("attack");
        if (!kinds.length) kinds.push("other");
        const fallbackPower = Math.max(0, Number(move.basePower || 0)) / 100;
        for (const targetLoc of targetLocs) {
          const suffix = targetLoc == null ? "" : ` ${targetLoc > 0 ? "+" : ""}${targetLoc}`;
          const targetCode = targetLoc == null ? 2 : targetLoc + 2;
          const actionIndex = 7 + 5 * i + targetCode;
          const command = `move ${requestMoveIndex + 1}${suffix}`;
          const action = {
            type: "move", move_id: move.id,
            target: targetLoc == null ? 0 : targetLoc,
            mega: false, terastallize: false,
            ...(forcedTargetless ? {forced_targetless: true} : {}),
          };
          const estimatedDamage = estimatedDamageScore(
            battle, sideIndex, activeIndex, move, targetLoc,
          );
          // A damaging move aimed at an ally must retain its negative estimate;
          // otherwise the generic base-power fallback would rank friendly fire
          // as highly as a hit on an opponent.
          const maxDamageScore = estimatedDamage.raw;
          const attackScore = estimatedDamage.shortlist < 0
            ? estimatedDamage.shortlist
            : Math.max(fallbackPower, estimatedDamage.shortlist);
          const tacticalScore = attackScore
            + (kinds.includes("protect") ? 0.15 : 0)
            + (kinds.includes("setup") ? 0.12 : 0);
          options.push({command, kinds, score: tacticalScore, maxDamageScore,
            action_index: actionIndex, action});
          // The target format can expose a battle gimmick in the request.
          // Keep each legal variant at poke-env's native action index.
          if (activeRequest.canMegaEvo) {
            options.push({
              command: `${command} mega`, kinds, score: tacticalScore,
              action_index: actionIndex + 20, maxDamageScore,
              action: {...action, mega: true},
            });
          }
          if (activeRequest.canTerastallize) {
            options.push({
              command: `${command} terastallize`, kinds, score: tacticalScore,
              action_index: actionIndex + 80, maxDamageScore,
              action: {...action, terastallize: true},
            });
          }
        }
      }
      const trapped = Boolean(activeRequest.trapped);
      if (!trapped) {
        for (let i = 0; i < request.side.pokemon.length; i++) {
          const pokemon = request.side.pokemon[i];
          if (!pokemon.active && !pokemon.condition.endsWith(" fnt")) {
            options.push({command: `switch ${i + 1}`, kinds: ["switch"], score: 0.15, action_index: nativeRosterIndex(battle, request, sideIndex, i) + 1, action: {type: "switch", species: Dex.species.get(String(pokemon.details || "").split(",")[0].trim()).id}});
          }
        }
      }
      return options;
    }

    function semanticActionKey(actions) {
      return JSON.stringify(actions.map(action => Object.fromEntries(
        Object.entries(action).sort(([left], [right]) => left.localeCompare(right)),
      )));
    }

    function generateJointCandidates(battle, sideIndex, preferredAction = null) {
      const candidateLimit = sideIndex === 0 ? maxCandidates : maxResponseCandidates;
      const request = requestForSide(battle, sideIndex);
      if (!request || request.wait) {
        throw new Error(`side ${sideIndex + 1} has no active choice request`);
      }
      const perSlot = [0, 1].map(slot => individualOptions(battle, request, sideIndex, slot));
      const candidates = [];
      const rejectionSamples = [];
      for (const first of perSlot[0]) {
        for (const second of perSlot[1]) {
          if (first.command.startsWith("switch ") && first.command === second.command) continue;
          const command = `${first.command}, ${second.command}`;
          const side = battle.sides[sideIndex];
          let complete = false;
          try {
            const valid = side.choose(command);
            complete = Boolean(valid && side.choice.actions.length === 2);
          } catch (_error) {
            complete = false;
            if (rejectionSamples.length < 8) {
              rejectionSamples.push({command, error: String(_error.message || _error)});
            }
          } finally {
            side.clearChoice();
          }
          if (!complete) {
            if (rejectionSamples.length < 8) {
              rejectionSamples.push({command, error: "choice was not complete"});
            }
            continue;
          }
          const kinds = [...new Set([...first.kinds, ...second.kinds])];
          candidates.push({
            command,
            kinds,
            score: first.score + second.score,
            maxDamageScore: Number(first.maxDamageScore || 0)
              + Number(second.maxDamageScore || 0),
            action_indices: [first.action_index, second.action_index],
            actions: [first.action, second.action],
          });
        }
      }
      const ranked = candidates.sort((a, b) => b.score - a.score || a.command.localeCompare(b.command));
      const selected = [];
      const useDamageRankedResponses = sideIndex === 1
        && responseSelectionStrategy === "damage_ranked";
      if (useDamageRankedResponses) {
        // Keep one direct-damage response, then use the tactical ranking for
        // the remaining columns. This avoids spending most of a narrow
        // response set on separate category reservations that can displace
        // several high-ranked joint actions.
        const damageResponses = [...candidates].sort((left, right) =>
          right.maxDamageScore - left.maxDamageScore
          || left.command.localeCompare(right.command)
        ).slice(0, Math.min(candidateLimit, maxDamageResponseCandidates));
        selected.push(...damageResponses);
        for (const candidate of ranked) {
          if (selected.length >= candidateLimit) break;
          if (!selected.includes(candidate)) selected.push(candidate);
        }
      }
      if (sideIndex === 1 && responseSelectionStrategy === "slot_coverage") {
        // Greedily cover high-ranked legal actions for both slots. The utility
        // decays with each slot's rank, so broad coverage cannot displace the
        // strongest actions merely to include many low-ranked alternatives.
        const optionRanks = perSlot.map(options => {
          const bestScoreByAction = new Map();
          for (const option of options) {
            const key = JSON.stringify(option.action);
            bestScoreByAction.set(key, Math.max(
              bestScoreByAction.get(key) ?? -Infinity,
              Number(option.score || 0),
            ));
          }
          return new Map([...bestScoreByAction.entries()]
            .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
            .map(([key], rank) => [key, 1 / (rank + 1)]));
        });
        const covered = [new Set(), new Set()];
        // Keep at least the strongest raw-damage joint response. Pure slot
        // coverage can spread the covered actions across incompatible joint
        // pairs and lose the response the opponent actually chooses.
        const topDamage = [...candidates].sort((left, right) =>
          right.maxDamageScore - left.maxDamageScore
          || left.command.localeCompare(right.command)
        ).slice(0, Math.min(candidateLimit, maxDamageResponseCandidates));
        for (const candidate of topDamage) {
          selected.push(candidate);
          candidate.actions.forEach((action, slot) => covered[slot].add(JSON.stringify(action)));
        }
        const maxJointScore = Math.max(1, ...ranked.map(row => Math.abs(row.score)));
        while (selected.length < candidateLimit) {
          let best = null;
          let bestUtility = -Infinity;
          for (const candidate of ranked) {
            if (selected.includes(candidate)) continue;
            const utility = candidate.actions.reduce((sum, action, slot) => {
              const key = JSON.stringify(action);
              return sum + (covered[slot].has(key) ? 0 : (optionRanks[slot].get(key) || 0));
            }, 0) + 0.05 * candidate.score / maxJointScore;
            if (utility > bestUtility) {
              best = candidate;
              bestUtility = utility;
            }
          }
          if (!best) break;
          selected.push(best);
          best.actions.forEach((action, slot) => covered[slot].add(JSON.stringify(action)));
        }
      }
      // Preserve a legal pass combination explicitly. This matters when one
      // active slot must switch or has no usable order: a switch-plus-pass
      // response can otherwise be dropped by the four-row joint-action cap.
      for (const kind of (useDamageRankedResponses
        || (responseSelectionStrategy === "slot_coverage" && sideIndex === 1)
        ? [] : ["pass", "switch", "protect", "setup", "redirection", "defensive"])) {
        if (selected.length >= candidateLimit) break;
        const candidate = ranked.find(
          row => !selected.includes(row) && row.kinds.includes(kind),
        );
        if (candidate) selected.push(candidate);
      }
      // Preserve high-damage responses in this public hidden-world hypothesis.
      // This knob permits a same-width ablation of damage-response coverage
      // against tactical diversity; it does not infer an opponent tendency.
      if (sideIndex === 1 && responseSelectionStrategy === "semantic") {
        const damageMaxCandidates = [...candidates].sort((a, b) =>
          b.maxDamageScore - a.maxDamageScore
          || a.command.localeCompare(b.command)
        ).slice(0, maxDamageResponseCandidates);
        const protectedDamageCandidates = new Set(damageMaxCandidates);
        for (const damageMaxCandidate of damageMaxCandidates) {
          if (selected.includes(damageMaxCandidate)) continue;
          if (selected.length >= candidateLimit) {
            const evictionIndex = selected.findLastIndex(
              candidate => !protectedDamageCandidates.has(candidate),
            );
            if (evictionIndex < 0) continue;
            selected.splice(evictionIndex, 1);
          }
          selected.push(damageMaxCandidate);
        }
      }
      if (!useDamageRankedResponses) {
        for (const candidate of ranked) {
          if (selected.length >= candidateLimit) break;
          if (!selected.includes(candidate)) selected.push(candidate);
        }
      }
      // Always retain the acting policy's proposal as a matrix row. Otherwise
      // a small heuristic shortlist can exclude the current policy entirely,
      // making search a replacement policy over an incomplete action space.
      if (sideIndex === 0 && Array.isArray(preferredAction) && preferredAction.length === 2) {
        const preferredKey = semanticActionKey(preferredAction);
        const preferredCandidate = candidates.find(
          candidate => semanticActionKey(candidate.actions) === preferredKey,
        );
        if (preferredCandidate && !selected.includes(preferredCandidate)) {
          if (selected.length >= candidateLimit) selected.pop();
          selected.push(preferredCandidate);
        }
      }
      selected.diagnostics = {
        perSlotOptionCounts: perSlot.map(options => options.length),
        generatedJointCandidateCount: candidates.length,
        rankedJointCandidates: sideIndex === 1 && returnResponseCandidates
          ? ranked.map((candidate, rank) => ({
            rank,
            actions: candidate.actions,
            score: candidate.score,
            max_damage_score: candidate.maxDamageScore,
            selected: selected.includes(candidate),
          }))
          : [],
        rejectionSamples,
      };
      return selected;
    }

    function previewCommand(indices) {
      if (!Array.isArray(indices) || indices.length !== 4 || new Set(indices).size !== 4) {
        throw new Error("each preview must list four distinct zero-based indices");
      }
      return `team ${indices.map(index => Number(index) + 1).join("")}`;
    }

    function materialValue(battle) {
      const totals = battle.sides.map(side => side.pokemon.reduce((sum, pokemon) => {
        if (!pokemon || pokemon.fainted || !pokemon.maxhp) return sum;
        return sum + pokemon.hp / pokemon.maxhp;
      }, 0));
      const advantage = (totals[0] - totals[1]) / 6;
      return 1 / (1 + Math.exp(-5 * advantage));
    }

    function observationForP1(battle) {
      const observation = Array(512).fill(0);
      observation[0] = Math.min(Math.max(Number(battle.turn || 0) / 100, 0), 1);
      const sides = battle.sides;
      const teams = sides.map(side => side.pokemon || []);
      const active = sides.map(side => side.active || []);
      const all = [...teams[0], ...teams[1]];
      // The sampled opponent team is simulator state, not agent observation.
      // In this first-turn search only an active or fainted opponent has had
      // its individual state publicly revealed. The six species themselves
      // are supplied separately as the public Team Preview.
      const publiclyKnown = [
        ...teams[0].map(() => true),
        ...teams[1].map(pokemon => Boolean(pokemon?.active || pokemon?.fainted)),
      ];
      for (let index = 0; index < Math.min(12, all.length); index++) {
        const pokemon = all[index];
        if (!publiclyKnown[index]) continue;
        observation[1 + index] = pokemon?.maxhp ? Number(pokemon.hp || 0) / pokemon.maxhp : 0;
        observation[13 + index] = pokemon?.fainted ? 0 : 1;
      }
      for (let index = 0; index < 2; index++) {
        const pokemon = active[0][index];
        if (pokemon) observation[25 + index] = pokemon.maxhp ? Number(pokemon.hp || 0) / pokemon.maxhp : 0;
      }
      const boostNames = ["atk", "def", "spa", "spd", "spe"];
      const activeBoth = [...active[0].slice(0, 2), ...active[1].slice(0, 2)];
      for (let slot = 0; slot < activeBoth.length; slot++) {
        const boosts = activeBoth[slot]?.boosts || {};
        for (let index = 0; index < boostNames.length; index++) {
          const value = Number(boosts[boostNames[index]] || 0);
          observation[32 + slot * 5 + index] = Math.max(-6, Math.min(6, value)) / 6;
        }
      }
      const ids = value => Object.keys(value || {}).map(name => name.toUpperCase().replace(/[_-]/g, ""));
      const weather = new Set(ids(battle.weather));
      observation.splice(52, 4,
        Number(weather.has("RAINDANCE") || weather.has("PRIMORDIALSEA")),
        Number(weather.has("SUNNYDAY") || weather.has("DESOLATELAND")),
        Number(weather.has("SANDSTORM")),
        Number(weather.has("HAIL") || weather.has("SNOW") || weather.has("SNOWSCAPE")));
      const fields = new Set(ids(battle.field));
      ["ELECTRICTERRAIN", "GRASSYTERRAIN", "MISTYTERRAIN", "PSYCHICTERRAIN"].forEach((name, i) => observation[56 + i] = Number(fields.has(name)));
      observation[60] = Number(fields.has("TRICKROOM"));
      observation[61] = Number(fields.has("GRAVITY"));
      const ownConditions = new Set(ids(sides[0].sideConditions));
      const foeConditions = new Set(ids(sides[1].sideConditions));
      observation[62] = Number(ownConditions.has("TAILWIND"));
      observation[63] = Number(foeConditions.has("TAILWIND"));
      const screenIds = ["REFLECT", "LIGHTSCREEN", "AURORAVEIL"];
      screenIds.forEach((name, i) => {
        observation[100 + i] = Number(ownConditions.has(name));
        observation[103 + i] = Number(foeConditions.has(name));
      });
      const statusIds = {brn: 1, frz: 2, par: 3, psn: 4, slp: 5, tox: 6};
      for (let i = 0; i < 6; i++) {
        observation[64 + i] = (statusIds[teams[0][i]?.status] || 0) / 6;
        if (publiclyKnown[6 + i]) {
          observation[94 + i] = (statusIds[teams[1][i]?.status] || 0) / 6;
        }
      }
      for (let i = 0; i < Math.min(12, all.length); i++) {
        if (!publiclyKnown[i]) continue;
        const effects = all[i]?.volatiles || {};
        const keys = Object.keys(effects).map(name => name.toLowerCase().replace(/[_-]/g, ""));
        const perishIndex = keys.findIndex(name => /^perish[0-3]$/.test(name));
        const offset = 106 + i * 3;
        observation[offset] = Number(perishIndex >= 0);
        observation[offset + 1] = perishIndex >= 0 ? Number(keys[perishIndex].slice(-1)) / 3 : 0;
        observation[offset + 2] = Number(Boolean(all[i]?.trapped) || keys.includes("trapped"));
      }
      for (let sideIndex = 0; sideIndex < 2; sideIndex++) {
        for (let slot = 0; slot < 2; slot++) {
          const activePokemon = active[sideIndex][slot];
          const rosterIndex = teams[sideIndex].indexOf(activePokemon);
          if (activePokemon && rosterIndex >= 0 && rosterIndex < 6) {
            observation[70 + sideIndex * 12 + slot * 6 + rosterIndex] = 1;
          }
        }
      }
      for (let index = 6; index < Math.min(12, all.length); index++) {
        if (publiclyKnown[index]) continue;
        const volatileOffset = 106 + index * 3;
        if ([
          observation[1 + index], observation[13 + index],
          observation[94 + index - 6], observation[volatileOffset],
          observation[volatileOffset + 1], observation[volatileOffset + 2],
        ].some(value => value !== 0)) {
          throw new Error("unrevealed opponent state leaked into search frontier");
        }
      }
      return observation;
    }

    function nativeActionMaskForP1(battle) {
      const masks = [Array(126).fill(0), Array(126).fill(0)];
      if (battle.ended) return masks.flat();
      const request = requestForSide(battle, 0);
      if (!request) return masks.flat();
      for (let slot = 0; slot < 2; slot++) {
        for (const option of individualOptions(battle, request, 0, slot)) {
          if (option.action_index >= 0 && option.action_index < 126) {
            masks[slot][option.action_index] = 1;
          }
        }
      }
      return masks.flat();
    }

    function publicHistoryLines(battle) {
      const events = new Set([
        "switch", "drag", "replace", "turn", "move", "-singleturn", "-fail", "-block",
      ]);
      return battle.log.filter(line => events.has(String(line).split("|")[1]));
    }

    function publicProtocolMessages(battle) {
      const privateEvents = new Set(["showteam", "teamsheet", "request"]);
      return battle.log.filter(line => !privateEvents.has(String(line).split("|")[1]));
    }

    function activeMapForP1(battle) {
      const map = Array(24).fill(0);
      const side = battle.sides[0];
      for (let slot = 0; slot < 2; slot++) {
        const rosterIndex = side.pokemon.indexOf(side.active[slot]);
        if (rosterIndex >= 0 && rosterIndex < 6) map[slot * 6 + rosterIndex] = 1;
      }
      return map;
    }

    function ownTeamForModel(team) {
      return team.map(pokemon => ({
        species: pokemon.species || pokemon.speciesId || pokemon.name,
        moves: pokemon.moves || [],
        item: pokemon.item || "",
        ability: pokemon.ability || "",
      }));
    }

    let p1CandidateGenerationNs = 0n;
    let p2CandidateGenerationNs = 0n;
    function createSnapshot(world, worldIndex) {
      const globalWorldIndex = worldStartIndex + worldIndex;
      let battle;
      if (initialSnapshots !== undefined) {
        battle = State.deserializeBattle(initialSnapshots[worldIndex]);
        if (battle.format.id !== searchFormatId) throw new Error("initial snapshot format mismatch");
        if (battle.ended || battle.requestState !== "move") {
          throw new Error("initial snapshot must be a live move request");
        }
      } else {
        battle = new Battle({
          formatid: searchFormatId,
          // All hidden worlds represent the same known acting-player state.
          // Keep initialization identical so the acting player's preview order,
          // active slots, and legal candidate list cannot drift with a hidden
          // opponent hypothesis. Turn randomness is independently reseeded for
          // each world/sample below, with common random numbers across actions.
          seed: [seed, seed + 1, seed + 2, seed + 3],
          strictChoices: true,
          p1: {name: "Search P1", team: world.own_team},
          p2: {name: "Search P2 hypothesis", team: world.opponent_team},
        });
        battle.makeChoices(previewCommand(world.own_preview), previewCommand(world.opponent_preview));
        if (battle.ended || battle.requestState !== "move") {
          throw new Error("team preview did not reach a move request");
        }
      }
      let phaseStarted = process.hrtime.bigint();
      const p1Candidates = generateJointCandidates(battle, 0, preferredOwnAction);
      p1CandidateGenerationNs += process.hrtime.bigint() - phaseStarted;
      phaseStarted = process.hrtime.bigint();
      const p2Candidates = generateJointCandidates(battle, 1);
      p2CandidateGenerationNs += process.hrtime.bigint() - phaseStarted;
      const p1Request = requestForSide(battle, 0);
      const p1Diagnostic = {
        requestState: battle.requestState,
        forceSwitch: p1Request?.forceSwitch || null,
        active: (p1Request?.active || []).map((entry, index) => ({
          slot: index,
          ident: p1Request?.side?.pokemon?.[index]?.ident,
          condition: p1Request?.side?.pokemon?.[index]?.condition,
          moves: (entry?.moves || []).map(move => ({
            id: move.id,
            pp: move.pp,
            disabled: move.disabled,
            requestTarget: move.target,
            dexTarget: Dex.moves.get(move.id).target,
          })),
          options: individualOptions(battle, p1Request, 0, index).map(option => option.command),
        })),
      };
      const p2Request = returnResponseCandidates ? requestForSide(battle, 1) : null;
      const p2Diagnostic = returnResponseCandidates ? {
        requestState: battle.requestState,
        forceSwitch: p2Request?.forceSwitch || null,
        active: (p2Request?.active || []).map((entry, index) => ({
          slot: index,
          condition: p2Request?.side?.pokemon?.[index]?.condition,
          moves: (entry?.moves || []).map(move => ({
            id: move.id,
            pp: move.pp,
            disabled: move.disabled,
          })),
          options: individualOptions(battle, p2Request, 1, index).map(option => option.command),
        })),
        generated_joint_candidate_count: p2Candidates.diagnostics?.generatedJointCandidateCount ?? null,
        per_slot_option_counts: p2Candidates.diagnostics?.perSlotOptionCounts ?? null,
        ranked_joint_candidates: p2Candidates.diagnostics?.rankedJointCandidates ?? [],
      } : null;
      const beforeObservation = observationForP1(battle);
      const previousActiveMap = activeMapForP1(battle);
      const snapshot = JSON.stringify(State.serializeBattle(battle));
      battle.destroy();
      return {
        snapshot,
        p1Candidates,
        p2Candidates,
        p1Diagnostic,
        p2Diagnostic,
        p1PreviousObservation: beforeObservation,
        p1PreviousActiveMap: previousActiveMap,
        worldIndex: globalWorldIndex,
      };
    }

    // The action indices are the native choices seen by the live adapter.
    // Command strings are request-local move-slot labels and may differ when
    // a hidden hypothesis causes Transform to expose a different move order.
    const candidateKey = candidate => JSON.stringify(candidate.actions);
    const candidateSignature = candidates => JSON.stringify(
      candidates.map(candidateKey).sort()
    );
    const snapshotBuildStarted = process.hrtime.bigint();
    const allSnapshots = worlds.map(createSnapshot);
    const snapshotCandidateGenerationMs = Number(
      process.hrtime.bigint() - snapshotBuildStarted
    ) / 1e6;
    const candidateFilterStarted = process.hrtime.bigint();
    // A world whose hypothesis gives the acting player a different visible
    // move menu is inconsistent with the observation and must not enter the
    // robust payoff matrix. This occurs when an allied Imposter Ditto copies
    // a different preview species in different hypotheses.
    const observedActionSignature = candidateSignature(allSnapshots[0].p1Candidates);
    const consistentIndexes = allSnapshots
      .map((context, index) => candidateSignature(context.p1Candidates) === observedActionSignature ? index : -1)
      .filter(index => index >= 0);
    const snapshots = consistentIndexes.map(index => allSnapshots[index]);
    const consistentWorlds = consistentIndexes.map(index => worlds[index]);
    const inconsistentWorldCount = worlds.length - snapshots.length;
    // A hidden-world hypothesis can change a transformed allied Pokémon's
    // copied move menu. The acting player must choose one action that is legal
    // under every world in the robust matrix, so retain only matching commands
    // with the same native action indices in all worlds.
    const otherWorldCandidates = snapshots.slice(1).map(context =>
      new Set(context.p1Candidates.map(candidateKey))
    );
    const ownCandidates = snapshots[0].p1Candidates.filter(candidate =>
      otherWorldCandidates.every(candidates => candidates.has(candidateKey(candidate)))
    );
    if (!ownCandidates.length) {
      const candidateCounts = allSnapshots.map(context => ({
        p1: context.p1Candidates.length,
        p2: context.p2Candidates.length,
        p1Candidates: context.p1Candidates.map(candidate => ({
          actions: candidate.actions,
          action_indices: candidate.action_indices,
        })),
        candidateDiagnostics: context.p1Candidates.diagnostics,
        request: context.p1Diagnostic,
      }));
      throw new Error(`no legal acting-player candidates: ${JSON.stringify(candidateCounts)}`);
    }
    const candidateConsistencyFilterMs = Number(
      process.hrtime.bigint() - candidateFilterStarted
    ) / 1e6;
    const worldReports = [];
    const frontierStates = [];
    const wallStart = process.hrtime.bigint();
    let rolloutCount = 0;
    const payoffSums = snapshots.map(context =>
      Array.from({length: ownCandidates.length}, () => new Array(context.p2Candidates.length).fill(0))
    );
    const payoffSquares = snapshots.map(context =>
      Array.from({length: ownCandidates.length}, () => new Array(context.p2Candidates.length).fill(0))
    );
    let actualChanceSamples = 0;
    let adaptiveStopReason = maxChanceSamples === chanceSamples ? "fixed_budget" : "maximum_budget";
    let rolloutJsonParseNs = 0n;
    let rolloutDeserializationNs = 0n;
    let turnAdvanceNs = 0n;
    let materialValueNs = 0n;
    let battleCleanupNs = 0n;

    function robustActionIntervals(sampleCount) {
      return ownCandidates.map((_, row) => {
        let robustMean = Infinity;
        let robustLower = Infinity;
        let robustUpper = Infinity;
        let robustSe = 0;
        for (let worldIndex = 0; worldIndex < snapshots.length; worldIndex++) {
          for (let column = 0; column < snapshots[worldIndex].p2Candidates.length; column++) {
            const mean = payoffSums[worldIndex][row][column] / sampleCount;
            let standardError = 1;
            if (sampleCount > 1) {
              const variance = Math.max(
                0,
                (payoffSquares[worldIndex][row][column]
                  - sampleCount * mean * mean) / (sampleCount - 1),
              );
              standardError = Math.sqrt(variance / sampleCount);
            }
            robustMean = Math.min(robustMean, mean);
            robustLower = Math.min(
              robustLower,
              Math.max(0, mean - confidenceMultiplier * standardError),
            );
            robustUpper = Math.min(
              robustUpper,
              Math.min(1, mean + confidenceMultiplier * standardError),
            );
            robustSe = Math.max(robustSe, standardError);
          }
        }
        return {
          mean: robustMean,
          lower: robustLower,
          upper: robustUpper,
          standard_error_bound: robustSe,
        };
      });
    }

    const minimumSamplesBeforeEarlyStop = Math.min(
      maxChanceSamples,
      Math.max(chanceSamples, maxChanceSamples >= 4 ? 4 : maxChanceSamples),
    );
    const rolloutLoopStarted = process.hrtime.bigint();
    for (let sample = 0; sample < maxChanceSamples; sample++) {
      for (const [worldIndex, context] of snapshots.entries()) {
        const globalWorldIndex = context.worldIndex;
        for (let row = 0; row < ownCandidates.length; row++) {
          for (let column = 0; column < context.p2Candidates.length; column++) {
            let phaseStarted = process.hrtime.bigint();
            const state = JSON.parse(context.snapshot);
            rolloutJsonParseNs += process.hrtime.bigint() - phaseStarted;
            // Reuse each world's sample seed across action pairs (common random
            // numbers) so action comparisons see comparable chance streams.
            state.prng = [seed + globalWorldIndex * 100000 + sample * 101, seed + 1, seed + 2, seed + 3];
            phaseStarted = process.hrtime.bigint();
            const battle = State.deserializeBattle(state);
            rolloutDeserializationNs += process.hrtime.bigint() - phaseStarted;
            try {
              phaseStarted = process.hrtime.bigint();
              battle.makeChoices(ownCandidates[row].command, context.p2Candidates[column].command);
              turnAdvanceNs += process.hrtime.bigint() - phaseStarted;
            } catch (error) {
              throw new Error(
                `search choice failed (world=${globalWorldIndex}, row=${row}, column=${column}, `
                + `p1=${ownCandidates[row].command}, p2=${context.p2Candidates[column].command}): `
                + `${error.message}; own_actions=${JSON.stringify(ownCandidates[row].actions)}; `
                + `own_requests=${JSON.stringify(context.p1Diagnostic.active)}`,
              );
            }
            phaseStarted = process.hrtime.bigint();
            const sampledValue = materialValue(battle);
            materialValueNs += process.hrtime.bigint() - phaseStarted;
            payoffSums[worldIndex][row][column] += sampledValue;
            payoffSquares[worldIndex][row][column] += sampledValue * sampledValue;
            if (returnFrontiers) {
              const world = consistentWorlds[worldIndex];
              const frontier = {
                world_index: globalWorldIndex,
                row_index: row,
                column_index: column,
                chance_sample_index: sample,
                own_team: ownTeamForModel(world.own_team),
                opponent_team: world.opponent_team.map(pokemon => ({
                  species: pokemon.species || pokemon.speciesId || pokemon.name,
                })),
                observation: observationForP1(battle),
                action_mask: nativeActionMaskForP1(battle),
                public_history_lines: publicHistoryLines(battle),
                previous_own_observation: context.p1PreviousObservation,
                previous_own_active_map: context.p1PreviousActiveMap,
                previous_own_action_indices: ownCandidates[row].action_indices,
                format: searchFormatId,
              };
              if (returnProtocolMessages) {
                frontier.public_protocol_messages = publicProtocolMessages(battle);
                frontier.p1_request = battle.ended
                  ? null
                  : requestForSide(battle, 0);
              }
              frontierStates.push(frontier);
            }
            phaseStarted = process.hrtime.bigint();
            battle.destroy();
            battleCleanupNs += process.hrtime.bigint() - phaseStarted;
            rolloutCount++;
          }
        }
      }
      actualChanceSamples++;
      if (
        maxChanceSamples > chanceSamples
        && actualChanceSamples >= minimumSamplesBeforeEarlyStop
        && actualChanceSamples < maxChanceSamples
      ) {
        const intervals = robustActionIntervals(actualChanceSamples);
        const bestMeanIndex = intervals.reduce(
          (best, row, index) => row.mean > intervals[best].mean ? index : best,
          0,
        );
        const bestLower = intervals[bestMeanIndex].lower;
        const strongestAlternativeUpper = Math.max(
          ...intervals.filter((_, index) => index !== bestMeanIndex).map(row => row.upper),
          -Infinity,
        );
        if (bestLower > strongestAlternativeUpper) {
          adaptiveStopReason = "robust_action_uncertainty_bands_separated";
          break;
        }
      }
    }
    const rolloutLoopElapsedMs = Number(
      process.hrtime.bigint() - rolloutLoopStarted
    ) / 1e6;

    for (const [worldIndex, context] of snapshots.entries()) {
      const payoff = payoffSums[worldIndex].map(row =>
        row.map(total => total / actualChanceSamples)
      );
      const payoffStandardError = payoff.map((payoffRow, row) => payoffRow.map((mean, column) => {
        if (actualChanceSamples < 2) return null;
        const variance = Math.max(
          0,
          (payoffSquares[worldIndex][row][column]
            - actualChanceSamples * mean * mean) / (actualChanceSamples - 1),
        );
        return Math.sqrt(variance / actualChanceSamples);
      }));
      worldReports.push({
        world_index: context.worldIndex,
        opponent_response_count: context.p2Candidates.length,
        ...(returnResponseCandidates ? {
          opponent_response_candidates: context.p2Candidates.map(candidate => ({
            actions: candidate.actions,
            action_indices: candidate.action_indices,
          })),
          opponent_response_request_diagnostic: context.p2Diagnostic,
        } : {}),
        payoff_matrix: payoff,
        payoff_standard_error: payoffStandardError,
        chance_sample_count: actualChanceSamples,
        worst_response_by_own_action: payoff.map(row => Math.min(...row)),
      });
    }
    const robustIntervals = robustActionIntervals(actualChanceSamples);
    const robustScores = ownCandidates.map((_, row) => Math.min(...worldReports.map(world => world.worst_response_by_own_action[row])));
    const bestIndex = robustScores.indexOf(Math.max(...robustScores));
    const elapsedMs = Number(process.hrtime.bigint() - wallStart) / 1e6;
    return {
      format: searchFormatId,
      method: "worst_case_hidden_world_and_legal_opponent_response",
      response_selection_strategy: responseSelectionStrategy,
      frontier: "Showdown material heuristic (HP remaining); learned value model not wired",
      selected_action_index: bestIndex,
      selected_action: ownCandidates[bestIndex],
      own_candidates: ownCandidates,
      robust_scores: robustScores,
      robust_score_intervals: robustIntervals,
      world_count: snapshots.length,
      requested_world_count: worlds.length,
      inconsistent_world_count: inconsistentWorldCount,
      world_reports: worldReports,
      ...(returnFrontiers ? {frontier_states: frontierStates} : {}),
      chance_samples: actualChanceSamples,
      candidate_damage_samples: candidateDamageSamples,
      minimum_chance_samples: chanceSamples,
      maximum_chance_samples: maxChanceSamples,
      confidence_multiplier: confidenceMultiplier,
      adaptive_stop_reason: adaptiveStopReason,
      simulated_turn_advances: rolloutCount,
      snapshot_candidate_generation_ms: snapshotCandidateGenerationMs,
      p1_candidate_generation_ms: Number(p1CandidateGenerationNs) / 1e6,
      p2_candidate_generation_ms: Number(p2CandidateGenerationNs) / 1e6,
      estimated_damage_scoring_ms: Number(estimatedDamageScoringNs) / 1e6,
      candidate_consistency_filter_ms: candidateConsistencyFilterMs,
      rollout_loop_elapsed_ms: rolloutLoopElapsedMs,
      rollout_json_parse_ms: Number(rolloutJsonParseNs) / 1e6,
      rollout_deserialization_ms: Number(rolloutDeserializationNs) / 1e6,
      turn_advance_ms: Number(turnAdvanceNs) / 1e6,
      material_value_ms: Number(materialValueNs) / 1e6,
      battle_cleanup_ms: Number(battleCleanupNs) / 1e6,
      elapsed_ms: elapsedMs,
      transitions_per_second: rolloutCount / (elapsedMs / 1000),
      logical_cpu_count: os.availableParallelism(),
    };
}

if (require.main === module) {
  const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
  input.on("line", line => {
    if (!line.trim()) return;
    try {
      const result = runSearch(JSON.parse(line));
      process.stdout.write(`${JSON.stringify(result)}\n`);
    } catch (error) {
      if (process.env.POKE_SEARCH_PERSISTENT === "1") {
        process.stdout.write(`${JSON.stringify({error: String(error), stack: error.stack || ""})}\n`);
      } else {
        process.stderr.write(`${error.stack || error}\n`);
        process.exitCode = 1;
      }
    }
  });
}

module.exports = {runSearch};
