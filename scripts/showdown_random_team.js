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

const seed = Number.parseInt(process.argv[2] || "1", 10);
const formatId = process.argv[3] || "gen9championsvgc2026regmb";
const format = Dex.formats.get(formatId);
if (!format || !format.exists) {
  throw new Error(`Unknown Showdown format: ${process.argv[3]}`);
}
const validator = TeamValidator.get(formatId);

for (let offset = 0; offset < 100; offset++) {
  const currentSeed = seed + offset;
  const builder = new RandomChampionsTeams(
    formatId,
    new PRNG([currentSeed, currentSeed + 1, currentSeed + 2, currentSeed + 3])
  );
  const team = builder.randomTeam();
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
  if (!validationErrors) {
    process.stdout.write(JSON.stringify(team));
    process.exit(0);
  }
}
throw new Error("Could not generate a team satisfying Champions item clause");
