"use strict";

/** Select the recipient-appropriate branch of Showdown's split protocol log. */
function projectLogForPlayer(lines, playerSide) {
  if (!Array.isArray(lines)) throw new TypeError("protocol log must be an array");
  if (playerSide !== null && !["p1", "p2"].includes(playerSide)) {
    throw new RangeError("playerSide must be p1, p2, or null for public-only output");
  }
  const result = [];
  for (let index = 0; index < lines.length; index++) {
    const line = String(lines[index] ?? "");
    const match = /^\|split\|(p[1-4])$/.exec(line);
    if (!match) {
      if (line) result.push(line);
      continue;
    }
    if (index + 2 >= lines.length) {
      throw new Error("truncated Showdown split log entry");
    }
    const secret = String(lines[index + 1] ?? "");
    const shared = String(lines[index + 2] ?? "");
    const selected = playerSide === match[1] ? secret : shared;
    if (selected) result.push(selected);
    index += 2;
  }
  return result;
}

module.exports = {projectLogForPlayer};
