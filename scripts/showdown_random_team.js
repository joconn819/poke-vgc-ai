// Generate a legal team with Showdown's own Champions random-team builder.
const path = require("path");

const showdownRoot = process.env.POKEMON_SHOWDOWN_PATH;
if (!showdownRoot) {
  throw new Error("POKEMON_SHOWDOWN_PATH must point to a Pokemon Showdown checkout");
}

const { Dex } = require(path.join(showdownRoot, "dist/sim"));
const { TeamValidator } = require(path.join(showdownRoot, "dist/sim/team-validator"));
const { PRNG } = require(path.join(showdownRoot, "dist/sim/prng"));
const { RandomChampionsTeams } = require(path.join(
  showdownRoot,
  "dist/data/random-battles/champions/teams"
));

// Fixed, intentionally unusual MC legal cores for stratified robustness runs.
// Keep these small and validate the completed six-Pokemon roster below so
// regulation, item, species, ability, and move restrictions remain authoritative.
const archetypeSets = {
  no_guard_fissure: [
    { species: "Machamp", ability: "No Guard", moves: ["Wide Guard", "Close Combat", "Knock Off", "Protect"] },
    { species: "Gardevoir", ability: "Trace", moves: ["Skill Swap", "Psychic", "Protect", "Dazzling Gleam"] },
    { species: "Snorlax", ability: "Immunity", moves: ["Fissure", "Body Slam", "Protect", "Crunch"] },
  ],
  perish_trap: [
    { species: "Gengar", ability: "Cursed Body", moves: ["Perish Song", "Mean Look", "Protect", "Shadow Ball"] },
    { species: "Primarina", ability: "Torrent", moves: ["Perish Song", "Protect", "Moonblast", "Sparkling Aria"] },
  ],
  trick_room: [
    { species: "Slowbro", ability: "Regenerator", moves: ["Trick Room", "Scald", "Slack Off", "Psychic"] },
    { species: "Torkoal", ability: "Drought", moves: ["Eruption", "Heat Wave", "Earth Power", "Protect"] },
    { species: "Snorlax", ability: "Thick Fat", moves: ["Curse", "Body Slam", "Crunch", "Protect"] },
  ],
  tailwind_offense: [
    { species: "Talonflame", ability: "Gale Wings", moves: ["Tailwind", "Brave Bird", "Flare Blitz", "Taunt"] },
    { species: "Garchomp", ability: "Rough Skin", moves: ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"] },
    { species: "Gyarados", ability: "Intimidate", moves: ["Waterfall", "Ice Fang", "Dragon Dance", "Protect"] },
  ],
  sand_balance: [
    { species: "Tyranitar", ability: "Sand Stream", moves: ["Rock Slide", "Crunch", "Low Kick", "Protect"] },
    { species: "Excadrill", ability: "Sand Rush", moves: ["High Horsepower", "Iron Head", "Rock Slide", "Protect"] },
    { species: "Garchomp", ability: "Rough Skin", moves: ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"] },
  ],
};
function generateTeam(seed, formatId = "gen9championsdoubles2026mcclosed", archetype = "random") {
  const format = Dex.formats.get(formatId);
  if (!format || !format.exists) throw new Error(`Unknown Showdown format: ${formatId}`);
  if (archetype !== "random" && !archetypeSets[archetype]) {
    throw new Error(`Unknown team archetype: ${archetype}`);
  }
  const validator = TeamValidator.get(formatId);
  for (let offset = 0; offset < 100; offset++) {
    const currentSeed = seed + offset;
    const builder = new RandomChampionsTeams(
      formatId,
      new PRNG([currentSeed, currentSeed + 1, currentSeed + 2, currentSeed + 3])
    );
    const team = builder.randomTeam();
    if (archetype !== "random") {
      for (const [index, set] of archetypeSets[archetype].entries()) {
        team[index] = {
          ...team[index],
          ...set,
          name: set.species,
          speciesId: set.species.toLowerCase().replace(/[^a-z0-9]/g, ""),
          item: "",
          level: 50,
          nature: "Serious",
          evs: { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 },
          ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
        };
      }
    }
    const usedItems = new Set();
    for (const set of team) {
      if (set.item && usedItems.has(set.item)) {
        // The random-battle builder does not enforce formats that add Item Clause.
        set.item = "";
      } else if (set.item) {
        usedItems.add(set.item);
      }
    }
    const validationErrors = validator.validateTeam(team);
    if (!validationErrors) return team;
  }
  throw new Error("Could not generate a team satisfying Champions item clause");
}

module.exports = { generateTeam };

if (require.main === module) {
  const seed = Number.parseInt(process.argv[2] || "1", 10);
  const formatId = process.argv[3] || "gen9championsdoubles2026mcclosed";
  const archetype = process.argv[4] || "random";
  process.stdout.write(JSON.stringify(generateTeam(seed, formatId, archetype)));
}
