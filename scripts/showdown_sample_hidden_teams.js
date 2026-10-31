#!/usr/bin/env node
"use strict";

// Sample legal hidden-set hypotheses conditioned only on visible species and
// observed set facts. This is a hypothesis generator, not an opponent model.
const path = require("path");
const root = process.env.POKEMON_SHOWDOWN_PATH;
if (!root) throw new Error("Set POKEMON_SHOWDOWN_PATH to the local Showdown checkout");
const { Dex, TeamValidator, toID } = require(path.join(root, "dist/sim"));
const { PRNG } = require(path.join(root, "dist/sim/prng"));
const { RandomChampionsTeams } = require(path.join(
  root, "dist/data/random-battles/champions/teams"
));

const formatId = process.argv[2] || "gen9championsdoubles2026mcclosed";
function sample(request) {
  const seed = Number(request.seed || 1);
  const count = Number(request.count || 1);
  try {
    const roster = request.preview_species;
    if (!Array.isArray(roster) || roster.length !== 6) {
      throw new Error("preview_species must contain six species");
    }
    if (!Number.isInteger(count) || count < 1) throw new Error("count must be positive");
    const format = Dex.formats.get(formatId);
    if (!format.exists || format.gameType !== "doubles") {
      throw new Error(`Expected a supported doubles format: ${formatId}`);
    }
    const validator = TeamValidator.get(formatId);
    const teams = [];
    const attemptsPerTeam = Number(request.attempts_per_team || 500);
    for (let teamIndex = 0; teamIndex < count; teamIndex++) {
      let accepted = null;
      let lastRejection = null;
      let rejectionCounts = {knownMoves: 0, conditionedMoves: 0, teamItemClause: 0, teamValidator: 0};
      for (let attempt = 0; attempt < attemptsPerTeam && !accepted; attempt++) {
        const prng = new PRNG([seed + teamIndex * attemptsPerTeam + attempt, seed + 1, seed + 2, seed + 3]);
        const builder = new RandomChampionsTeams(formatId, prng);
        const team = [];
        for (const name of roster) {
          const species = Dex.species.get(name);
          // Cosmetic formes such as Furfrou-Dandy share the base species'
          // random-set entry. Preserve the previewed forme on the set while
          // asking the format builder for that base entry.
          const setSpecies = builder.randomDoublesSets[species.id]
            ? species
            : Dex.species.get(species.baseSpecies);
          const set = builder.randomSet(setSpecies, {}, true, true);
          if (setSpecies.id !== species.id) {
            set.species = species.name;
            set.speciesId = species.id;
          }
          const facts = request.revealed_facts?.[species.id] || {};
          // Struggle is a battle-time fallback when all PP is exhausted; it
          // is never part of the Pokemon's hidden set and is not learnable.
          const knownMoves = [...new Set((facts.moves || []).map(toID).filter(move => move !== "struggle"))];
          if (knownMoves.length > 4) {
            rejectionCounts.knownMoves++;
            lastRejection = {
              kind: "known_moves",
              species: species.name,
              required: knownMoves,
              reason: "more than four distinct moves were publicly revealed",
            };
            team.length = 0;
            break;
          }
          if (facts.moves_exact) {
            const exactMoves = knownMoves.map(move => Dex.moves.get(move));
            if (exactMoves.length !== 4 || exactMoves.some(move => !move.exists)) {
              rejectionCounts.knownMoves++;
              lastRejection = {
                kind: "known_moves",
                species: species.name,
                required: knownMoves,
                reason: "an exact copied move set must contain four valid moves",
              };
              team.length = 0;
              break;
            }
            set.moves = exactMoves.map(move => move.name);
          }
          // Random-set samples are a useful prior but can omit a perfectly
          // legal revealed move (for example a move from a revealed Mega
          // form). Condition the sample by replacing unobserved moves, then
          // let TeamValidator reject any resulting illegal combination.
          const moveIds = set.moves.map(toID);
          const protectedMoves = new Set(knownMoves);
          let incompatibleKnownMoves = false;
          for (const move of (facts.moves_exact ? [] : knownMoves)) {
            if (moveIds.includes(move)) continue;
            const replacementIndex = moveIds.findIndex(candidate => !protectedMoves.has(candidate));
            if (replacementIndex < 0) {
              rejectionCounts.knownMoves++;
              lastRejection = {
                kind: "known_moves",
                species: species.name,
                required: knownMoves,
                sampled: moveIds,
                reason: "no unobserved move slot remained",
              };
              team.length = 0;
              incompatibleKnownMoves = true;
              break;
            }
            moveIds[replacementIndex] = move;
            set.moves[replacementIndex] = Dex.moves.get(move).name;
            rejectionCounts.conditionedMoves++;
          }
          if (incompatibleKnownMoves) break;
          // These are observed facts, so sample from the conditioned set space
          // directly instead of waiting for the random-set builder to draw a
          // rare known value (notably a Mega Stone). TeamValidator below still
          // rejects values that are illegal for the species or format.
          if (facts.item) set.item = Dex.items.get(facts.item).name;
          if (facts.ability) set.ability = Dex.abilities.get(facts.ability).name;
          if (facts.tera_type) set.teraType = Dex.types.get(facts.tera_type).name;
          team.push(set);
        }
        if (team.length !== 6) continue;
        // The random-set helper samples sets independently; apply the format's
        // Item Clause the same way the full random-team path does.
        const itemOwner = new Map();
        for (const set of team) {
          const item = toID(set.item);
          if (!item) continue;
          const prior = itemOwner.get(item);
          if (!prior) {
            itemOwner.set(item, set);
            continue;
          }
          const priorKnown = Boolean(request.revealed_facts?.[toID(prior.species)]?.item);
          const currentKnown = Boolean(request.revealed_facts?.[toID(set.species)]?.item);
          if (priorKnown && currentKnown) {
            rejectionCounts.teamItemClause++;
            lastRejection = {
              kind: "duplicate_revealed_item",
              item,
              species: [prior.species, set.species],
            };
            team.length = 0;
            break;
          }
          if (currentKnown) {
            prior.item = "";
            itemOwner.set(item, set);
          } else {
            set.item = "";
          }
        }
        if (team.length !== 6) continue;
        const errors = validator.validateTeam(team);
        if (!errors) {
          accepted = team;
        } else {
          rejectionCounts.teamValidator++;
          lastRejection = {
            kind: "team_validator",
            errors: errors.map(error => String(error.message || error)),
          };
        }
      }
      if (!accepted) {
        throw new Error(
          `Could not sample a legal hypothesis matching public facts for world ${teamIndex + 1}; `
          + `attempts=${attemptsPerTeam}; rejection_counts=${JSON.stringify(rejectionCounts)}; `
          + `last_rejection=${JSON.stringify(lastRejection)}`,
        );
      }
        teams.push(accepted);
    }
    return {format: formatId, seed, teams};
  } catch (error) {
    if (process.env.POKE_SEARCH_PERSISTENT === "1") {
      return {error: String(error), stack: error.stack || ""};
    }
    throw error;
  }
}

if (process.env.POKE_SEARCH_PERSISTENT === "1") {
  const readline = require("readline");
  const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
  input.on("line", line => {
    if (!line.trim()) return;
    try {
      process.stdout.write(`${JSON.stringify(sample(JSON.parse(line)))}\n`);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({error: String(error), stack: error.stack || ""})}\n`);
    }
  });
} else {
  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", chunk => input += chunk);
  process.stdin.on("end", () => {
    try {
      const request = JSON.parse(input);
      process.stdout.write(JSON.stringify(sample({...request, seed: Number(process.argv[3] || 1), count: Number(process.argv[4] || 1)})));
    } catch (error) {
      process.stderr.write(`${error.stack || error}\n`);
      process.exitCode = 1;
    }
  });
}
