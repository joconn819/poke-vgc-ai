#!/usr/bin/env node
"use strict";

const path = require("node:path");
const readline = require("node:readline");

const showdownRoot = process.argv[2];
if (!showdownRoot) throw new Error("usage: node ninjask_showdown_bridge.js SHOWDOWN_ROOT");
const { Battle } = require(path.join(showdownRoot, "dist/sim"));
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

function p1Log(log) {
  const out = [];
  for (let i = 0; i < log.length; i++) {
    const line = log[i];
    if (line.startsWith("|split|")) {
      const side = line.split("|")[2];
      out.push(side === "p1" ? log[i + 1] : log[i + 2]);
      i += 2;
      continue;
    }
    if (line.startsWith("|t:|") || line.startsWith("|gen|") || line.startsWith("|rule|") ||
        line === "|" || line.startsWith("|-hint|")) continue;
    out.push(line);
  }
  return out;
}

rl.on("line", (line) => {
  const input = JSON.parse(line);
  const battle = new Battle({
    formatid: input.format,
    seed: input.seed,
    strictChoices: true,
    p1: { name: "Player 1", team: input.team1 },
    p2: { name: "Player 2", team: input.team2 },
  });
  let error = null;
  let at = -1;
  for (let i = 0; i < input.choices.length; i++) {
    try {
      const [p1, p2] = input.choices[i];
      battle.makeChoices(p1, p2);
    } catch (e) {
      error = String(e.message || e);
      at = i;
      break;
    }
  }
  process.stdout.write(JSON.stringify({
    ended: battle.ended,
    turn: battle.turn,
    winner: battle.winner,
    log: p1Log(battle.log),
    error,
    at,
  }) + "\n");
});
