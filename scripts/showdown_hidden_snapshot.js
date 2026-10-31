"use strict";

const path = require("path");
const showdownRoot = process.env.POKEMON_SHOWDOWN_PATH;
if (!showdownRoot) throw new Error("Set POKEMON_SHOWDOWN_PATH to the local Showdown checkout");
const {Dex} = require(path.join(showdownRoot, "dist/sim"));

/**
 * Replace only opponent Pokemon that have never been in battle with the
 * corresponding entries from a public-compatible hidden-team hypothesis.
 * The input snapshots must be from the same format and roster order. A
 * sampled bench species may differ from the live hidden species only when it
 * appears in the six-species public preview. Revealed/active Pokemon are
 * deliberately rejected: use the belief-conditioned runtime reconstruction.
 */
function decodeSnapshot(value) {
  const state = typeof value === "string" ? JSON.parse(value) : structuredClone(value);
  if (!state || !Array.isArray(state.sides) || state.sides.length !== 2) {
    throw new TypeError("expected a serialized two-side Showdown battle");
  }
  return state;
}

function toId(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function speciesValueId(value) {
  if (value && typeof value === "object") return toId(value.id || value.name);
  // Showdown's serialized battle state encodes Species references using
  // their inspect string (for example, "[Species:gardevoir]") rather than
  // the public species name. Normalize that representation before Dex lookup.
  const serializedSpecies = /^\[Species:([^\]]+)\]$/i.exec(String(value || ""));
  return toId(serializedSpecies ? serializedSpecies[1] : value);
}

function pokemonSpeciesIds(pokemon, {includeSet = false} = {}) {
  const values = [pokemon?.species, pokemon?.baseSpecies];
  if (includeSet) values.push(pokemon?.set?.species);
  return [...new Set(values.map(speciesValueId).filter(Boolean))];
}

function matchesSpeciesId(pokemon, speciesId, {includeSet = false} = {}) {
  const normalizedSpeciesId = speciesValueId(speciesId);
  const expected = Dex.species.get(normalizedSpeciesId);
  const expectedId = toId(expected.id || normalizedSpeciesId);
  const expectedBaseId = toId(expected.baseSpecies || expected.name || normalizedSpeciesId);
  return pokemonSpeciesIds(pokemon, {includeSet}).some(candidateId => {
    if (candidateId === expectedId || candidateId === expectedBaseId) return true;
    const candidate = Dex.species.get(candidateId);
    return toId(candidate.baseSpecies || candidate.name) === expectedBaseId;
  });
}

function matchesPreviewSpecies(pokemon, previewSpecies) {
  return previewSpecies.some(species => matchesSpeciesId(pokemon, species, {includeSet: true}));
}

function replaceUnrevealedBenchPokemon(
  liveSnapshot,
  hypothesisSnapshot,
  sideIndex,
  rosterPositions,
  {publicPreviewSpecies = []} = {},
) {
  if (![0, 1].includes(sideIndex)) throw new RangeError("sideIndex must be 0 or 1");
  if (!Array.isArray(rosterPositions) || rosterPositions.length === 0) {
    throw new TypeError("rosterPositions must be a non-empty array");
  }
  const live = decodeSnapshot(liveSnapshot);
  const hypothesis = decodeSnapshot(hypothesisSnapshot);
  if (live.formatid !== hypothesis.formatid || live.gameType !== hypothesis.gameType) {
    throw new Error("live and hypothesis snapshots use different battle formats");
  }
  const liveSide = live.sides[sideIndex];
  const hypothesisSide = hypothesis.sides[sideIndex];
  if (liveSide.pokemon.length !== hypothesisSide.pokemon.length) {
    throw new Error("live and hypothesis teams have different sizes");
  }
  const uniquePositions = [...new Set(rosterPositions.map(Number))];
  if (uniquePositions.length !== rosterPositions.length) {
    throw new Error("rosterPositions must not contain duplicates");
  }
  for (const position of uniquePositions) {
    if (!Number.isInteger(position) || position < 0 || position >= liveSide.pokemon.length) {
      throw new RangeError(`invalid roster position ${position}`);
    }
    const liveIndex = liveSide.pokemon.findIndex(pokemon => pokemon.position === position);
    const hypothesisIndex = hypothesisSide.pokemon.findIndex(pokemon => pokemon.position === position);
    if (liveIndex < 0 || hypothesisIndex < 0) {
      throw new Error(`roster position ${position} is missing from a snapshot`);
    }
    const current = liveSide.pokemon[liveIndex];
    const sampled = hypothesisSide.pokemon[hypothesisIndex];
    if (current.isActive || current.activeTurns > 0 || current.fainted) {
      throw new Error(`cannot replace revealed Pokemon at roster position ${position}`);
    }
    if (sampled.isActive || sampled.activeTurns > 0 || sampled.fainted) {
      throw new Error(`hypothesis Pokemon at roster position ${position} is not an untouched bench member`);
    }
    if (!matchesPreviewSpecies(sampled, publicPreviewSpecies)) {
      throw new Error(
        `hypothesis species is outside the public preview at roster position ${position}`,
      );
    }
    // The complete entry comes from the hypothesis snapshot. This replaces
    // move slots, item, ability, stats, and all other set-derived fields, so
    // the true unseen set is not carried into the deserialized branch.
    liveSide.pokemon[liveIndex] = structuredClone(sampled);
  }
  return live;
}

/**
 * Align a sampled opponent roster with a live state whose Pokemon rows have
 * been reordered by switches. `publicSpeciesByPosition` may name only active,
 * fainted, or previously active Pokemon. Untouched hidden bench rows are
 * assigned to remaining hypothesis slots without reading their live sets.
 */
function alignHypothesisRosterToLive(
  liveSnapshot,
  hypothesisSnapshot,
  sideIndex,
  {publicSpeciesByPosition = {}} = {},
) {
  if (![0, 1].includes(sideIndex)) throw new RangeError("sideIndex must be 0 or 1");
  const live = decodeSnapshot(liveSnapshot);
  const hypothesis = decodeSnapshot(hypothesisSnapshot);
  if (live.formatid !== hypothesis.formatid || live.gameType !== hypothesis.gameType) {
    throw new Error("live and hypothesis snapshots use different battle formats");
  }
  const liveSide = live.sides[sideIndex];
  const hypothesisSide = hypothesis.sides[sideIndex];
  if (liveSide.pokemon.length !== hypothesisSide.pokemon.length) {
    throw new Error("live and hypothesis teams have different sizes");
  }
  const speciesToTarget = new Map();
  const baseSpeciesToTargets = new Map();
  for (const pokemon of hypothesisSide.pokemon) {
    const speciesId = toId(pokemon.set?.species || pokemon.species);
    if (!speciesId || speciesToTarget.has(speciesId)) {
      throw new Error("hypothesis roster species must be unique and named");
    }
    speciesToTarget.set(speciesId, Number(pokemon.position));
    const hypothesisSpecies = pokemon.set?.species || pokemon.species;
    const baseSpeciesId = toId(Dex.species.get(hypothesisSpecies).baseSpecies || hypothesisSpecies);
    if (baseSpeciesId) {
      const targets = baseSpeciesToTargets.get(baseSpeciesId) || [];
      targets.push(Number(pokemon.position));
      baseSpeciesToTargets.set(baseSpeciesId, targets);
    }
  }
  const targetToLive = new Map();
  const usedLive = new Set();
  for (const [rawPosition, species] of Object.entries(publicSpeciesByPosition)) {
    const oldPosition = Number(rawPosition);
    const current = liveSide.pokemon.find(row => Number(row.position) === oldPosition);
    if (!current) throw new RangeError(`public roster position ${oldPosition} is missing`);
    if (!(current.isActive || current.activeTurns > 0 || current.fainted)) {
      throw new Error(`cannot map unrevealed Pokemon at roster position ${oldPosition}`);
    }
    // publicSpeciesByPosition is built exclusively from public switch and
    // replacement protocol events. Do not compare it with serialized live
    // Pokemon species/baseSpecies here: those fields can contain hidden
    // identity during Transform or Illusion.
    const publicSpecies = Dex.species.get(species);
    const publicSpeciesId = toId(publicSpecies.id || species);
    const publicBaseSpeciesId = toId(publicSpecies.baseSpecies || species);
    const targetRosterPosition = speciesToTarget.get(publicSpeciesId)
      ?? (baseSpeciesToTargets.get(publicBaseSpeciesId)?.length === 1
        ? baseSpeciesToTargets.get(publicBaseSpeciesId)[0]
        : undefined);
    if (targetRosterPosition === undefined) {
      throw new Error(`hypothesis omits publicly revealed species ${species}`);
    }
    if (targetToLive.has(targetRosterPosition)) {
      throw new Error("public species map has duplicate hypothesis species");
    }
    targetToLive.set(targetRosterPosition, oldPosition);
    usedLive.add(oldPosition);
  }
  const remainingTargets = hypothesisSide.pokemon
    .map(row => Number(row.position))
    .filter(position => !targetToLive.has(position));
  const remainingSources = liveSide.pokemon
    .map(row => Number(row.position))
    .filter(position => !usedLive.has(position));
  if (remainingTargets.length !== remainingSources.length) {
    throw new Error("roster remap did not account for every Pokemon");
  }
  remainingTargets.forEach((target, index) => targetToLive.set(target, remainingSources[index]));

  const alignedHypothesisPokemon = new Array(hypothesisSide.pokemon.length);
  for (const pokemon of hypothesisSide.pokemon) {
    const targetPosition = Number(pokemon.position);
    const livePosition = targetToLive.get(targetPosition);
    if (livePosition === undefined) {
      throw new Error(`hypothesis position ${targetPosition} is unmapped`);
    }
    const copy = structuredClone(pokemon);
    copy.position = livePosition;
    alignedHypothesisPokemon[livePosition] = copy;
  }
  if (alignedHypothesisPokemon.some(row => !row)) {
    throw new Error("hypothesis alignment produced an empty live roster position");
  }
  hypothesisSide.pokemon = alignedHypothesisPokemon;
  liveSide.team = [...speciesToTarget.values()]
    .sort((left, right) => left - right)
    .map(targetPosition => Number(targetToLive.get(targetPosition)) + 1)
    .join("");
  return {liveSnapshot: live, hypothesisSnapshot: hypothesis};
}

const PUBLIC_RUNTIME_FIELDS = [
  "position", "details", "gender", "status", "statusState", "volatiles",
  "boosts", "fainted", "faintQueued", "subFainted", "formeRegression",
  "types", "baseTypes", "addedType", "knownType", "apparentType",
  "switchFlag", "forceSwitchFlag", "skipBeforeSwitchOutEventFlag",
  "draggedIn", "newlySwitched", "beingCalledBack", "lastMove",
  "lastMoveEncore", "lastMoveUsed", "moveThisTurn", "statsRaisedThisTurn",
  "statsLoweredThisTurn", "hurtThisTurn", "lastDamage", "attackedBy",
  "timesAttacked", "isActive", "activeTurns", "activeMoveActions",
  "previouslySwitchedIn", "truantTurn", "bondTriggered",
  "heroMessageDisplayed", "swordBoost", "shieldBoost", "syrupTriggered",
  "stellarBoostedTypes", "isStarted", "duringMove", "terastallized",
  "canMegaEvo", "canUltraBurst", "canGigantamax", "canTerastallize",
  "lastItem", "usedItemThisTurn", "ateBerry", "itemKnockedOff", "trapped",
  "maybeTrapped", "maybeDisabled", "maybeLocked", "illusion", "transformed",
];

/**
 * Reconstruct a revealed opponent Pokemon from a sampled set and the public
 * runtime facts observed in the live battle. `publicHpFraction` must come from
 * the player-view protocol/observation, not the simulator's exact hidden HP.
 * Quantize it first if the protocol reports rounded percentages.
 * `revealedMoveIds` contains only moves present in the public battle history.
 */
function replaceRevealedPokemonWithHypothesis(
  liveSnapshot,
  hypothesisSnapshot,
  sideIndex,
  rosterPosition,
  {
    publicHpFraction,
    publicSpecies,
    revealedMoveIds = [],
    revealedItem,
    revealedAbility,
    activeAbility,
    revealedTeraType,
    teraTypeRevealed = false,
  } = {},
) {
  if (![0, 1].includes(sideIndex)) throw new RangeError("sideIndex must be 0 or 1");
  if (!Number.isInteger(rosterPosition) || rosterPosition < 0) {
    throw new RangeError("rosterPosition must be a non-negative integer");
  }
  if (!Number.isFinite(publicHpFraction) || publicHpFraction < 0 || publicHpFraction > 1) {
    throw new RangeError("publicHpFraction must be a finite value in [0, 1]");
  }
  if (!Array.isArray(revealedMoveIds)) throw new TypeError("revealedMoveIds must be an array");
  const live = decodeSnapshot(liveSnapshot);
  const hypothesis = decodeSnapshot(hypothesisSnapshot);
  if (live.formatid !== hypothesis.formatid || live.gameType !== hypothesis.gameType) {
    throw new Error("live and hypothesis snapshots use different battle formats");
  }
  const liveSide = live.sides[sideIndex];
  const hypothesisSide = hypothesis.sides[sideIndex];
  const liveIndex = liveSide.pokemon.findIndex(pokemon => pokemon.position === rosterPosition);
  const hypothesisIndex = hypothesisSide.pokemon.findIndex(pokemon => pokemon.position === rosterPosition);
  if (liveIndex < 0 || hypothesisIndex < 0) {
    throw new Error(`roster position ${rosterPosition} is missing from a snapshot`);
  }
  const current = liveSide.pokemon[liveIndex];
  const sampled = structuredClone(hypothesisSide.pokemon[hypothesisIndex]);
  const hypothesisMoveSlots = structuredClone(sampled.moveSlots);
  if (!publicSpecies) {
    throw new Error("revealed Pokemon reconstruction requires a public species identity");
  }
  const observedSpecies = publicSpecies;
  if (!matchesSpeciesId(sampled, observedSpecies, {includeSet: true})) {
    throw new Error(
      `hypothesis species mismatch at roster position ${rosterPosition} `
      + `(public species ${observedSpecies}, sampled species ${sampled.set?.species || sampled.species})`,
    );
  }
  if (revealedItem != null && toId(sampled.set?.item) !== toId(revealedItem)) {
    throw new Error(`hypothesis item contradicts the revealed item at roster position ${rosterPosition}`);
  }
  if (revealedAbility != null && toId(sampled.set?.ability) !== toId(revealedAbility)) {
    throw new Error(`hypothesis ability contradicts the revealed ability at roster position ${rosterPosition}`);
  }
  if (revealedTeraType != null && toId(sampled.set?.teraType) !== toId(revealedTeraType)) {
    throw new Error(`hypothesis Tera type contradicts the revealed type at roster position ${rosterPosition}`);
  }
  const sampledMoves = new Map(sampled.moveSlots.map(move => [move.id, move]));
  const currentMoves = new Map(current.moveSlots.map(move => [move.id, move]));
  for (const moveId of new Set(revealedMoveIds)) {
    if (!sampledMoves.has(moveId)) {
      throw new Error(`hypothesis omits publicly revealed move ${moveId}`);
    }
    const observed = currentMoves.get(moveId);
    if (observed) {
      const replacement = sampledMoves.get(moveId);
      for (const key of ["pp", "disabled", "disabledSource", "used"]) {
        if (key in observed) replacement[key] = observed[key];
      }
    }
  }
  for (const key of PUBLIC_RUNTIME_FIELDS) {
    if (key in current) sampled[key] = structuredClone(current[key]);
  }
  // A publicly observed temporary/copy ability belongs to the active battle
  // state. Keep it on the simulated Pokemon while leaving sampled.set.ability
  // as a hypothesis about the hidden base set.
  if (activeAbility != null) sampled.ability = toId(activeAbility);
  // Transform and non-base-form states are skipped by the worker using only
  // public protocol events. Never copy serialized species/baseSpecies fields
  // from the live opponent Pokemon into a search branch.
  if ((teraTypeRevealed || current.terastallized || revealedTeraType != null) && "teraType" in current) {
    sampled.teraType = current.teraType;
  }
  // Item consumption/knock-off is public dynamic state. The sampled item is
  // retained otherwise, so an unrevealed actual item cannot leak into search.
  if (
    revealedItem != null
    && (toId(current.lastItem) === toId(revealedItem) || current.itemKnockedOff)
  ) sampled.item = "";
  // Preserve only publicly visible ability suppression.
  if (current.volatiles?.gastroacid || current.volatiles?.coreenforcer) {
    sampled.ability = "";
  }
  const hpFraction = publicHpFraction;
  sampled.hp = current.fainted
    ? 0
    // The target Champions format publishes floor(100 * HP / maxHP). Keep
    // the simulated branch at the top of the observed percentage interval so
    // a sampled max HP does not round down to a different public percentage.
    : Math.min(sampled.maxhp, Math.max(1, Math.ceil(hpFraction * sampled.maxhp)));
  sampled.fainted = Boolean(current.fainted);
  liveSide.pokemon[liveIndex] = sampled;
  return live;
}

/** Build one complete simulator branch using only a sampled opponent world. */
function buildPublicCompatibleSnapshot(
  liveSnapshot,
  hypothesisSnapshot,
  sideIndex,
  {publicPreviewSpecies, publicStateByPosition = {}} = {},
) {
  const live = decodeSnapshot(liveSnapshot);
  const side = live.sides[sideIndex];
  if (!side || !Array.isArray(side.pokemon)) {
    throw new RangeError("sideIndex does not identify a battle side");
  }
  const revealed = [];
  const untouchedBench = [];
  for (const pokemon of side.pokemon) {
    const position = Number(pokemon.position);
    if (pokemon.isActive || pokemon.activeTurns > 0 || pokemon.fainted) {
      if (!publicStateByPosition[position]) {
        throw new Error(`missing player-view facts for revealed roster position ${position}`);
      }
      revealed.push(position);
    } else {
      untouchedBench.push(position);
    }
  }
  let result = live;
  if (untouchedBench.length) {
    if (!Array.isArray(publicPreviewSpecies)) {
      throw new TypeError("publicPreviewSpecies is required for untouched bench hypotheses");
    }
    result = replaceUnrevealedBenchPokemon(
      result,
      hypothesisSnapshot,
      sideIndex,
      untouchedBench,
      {publicPreviewSpecies},
    );
  }
  for (const position of revealed) {
    result = replaceRevealedPokemonWithHypothesis(
      result,
      hypothesisSnapshot,
      sideIndex,
      position,
      publicStateByPosition[position],
    );
  }
  return result;
}

module.exports = {
  buildPublicCompatibleSnapshot,
  alignHypothesisRosterToLive,
  replaceUnrevealedBenchPokemon,
  replaceRevealedPokemonWithHypothesis,
};
