#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const showdownPath = path.resolve(option(
  "--showdown-path",
  "/home/justin/p/poke_learning/pokemon-showdown"
));
const inputPath = path.resolve(option(
  "--input",
  "training/outputs/phase12_headless/round594_competitive_event_splits_20261113/teams_by_event.json"
));
const format = option("--format", "gen9championsdoubles2026mcclosed");
const { TeamValidator } = require(path.join(showdownPath, "dist/sim/team-validator"));
const validator = new TeamValidator(format);
const data = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const failures = [];
let total = 0;
for (const [split, teams] of Object.entries(data)) {
  for (const row of teams) {
    total += 1;
    const problems = validator.validateTeam(row.team);
    if (problems) failures.push({ split, source_team_id: row.source_team_id, event: row.event, problems });
  }
}
const result = {
  status: failures.length ? "invalid" : "valid",
  format,
  input: inputPath,
  showdown_path: showdownPath,
  teams_checked: total,
  valid_teams: total - failures.length,
  invalid_teams: failures.length,
  failures,
};
console.log(JSON.stringify(result, null, 2));
if (failures.length) process.exitCode = 1;
