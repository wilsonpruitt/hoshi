// Bot strength / behaviour gate for the in-page bot (the `bot-kernel` block).
//   node web/test_bot.js          (HOSHI_FAST=1 for a quick smoke check)
// Loads the same engine + bot the browser runs, then checks the things players
// complained were missing: the bot ATTACKS (takes a capture that's there),
// DEFENDS (saves its own trooper from atari), and that self-play is aggressive
// (games actually end by decapitation, not only passive territory).
const fs = require('fs');
const path = require('path');
const H = require('./hoshi-rules.js');

// pull the DOM-free bot kernel out of hoshi.html and bind it to the engine
const html = fs.readFileSync(path.join(__dirname, 'hoshi.html'), 'utf8');
function botKernel() {
  const open = html.indexOf('<script id="bot-kernel"');
  const start = html.indexOf('>', open) + 1;
  const end = html.indexOf('</script>', start);
  const mod = { exports: {} };
  new Function('Hoshi', 'module', 'exports', html.slice(start, end))(H, mod, mod.exports);
  return mod.exports;
}
const Bot = botKernel();

let ok = 0, fail = 0;
const A = (c, m) => { if (c) ok++; else { fail++; console.log('  FAIL:', m); } };
const N = 9, idx = (r, c) => H.idx(N, r, c);
const FAST = !!process.env.HOSHI_FAST;

// helper: a fresh empty-ish board we can hand-place stones onto
function blank() { const s = H.newState(N, 'beachhead'); s.reserve = [5, 5]; s.toMove = 0; return s; }

// 1) ATTACK — an enemy trooper in atari must be captured (greedy AND mcts)
function ataris() {
  const s = blank(); const c = idx(4, 4);
  s.board[c] = { c: 1, k: 'T' }; s.placedPly[c] = -10;
  s.board[idx(3, 4)] = { c: 0, k: 'D' }; s.board[idx(5, 4)] = { c: 0, k: 'D' }; s.board[idx(4, 3)] = { c: 0, k: 'D' };
  return { s, c };               // the one empty liberty is (4,5)
}
{
  let r = ataris(); let ns = H.apply(r.s, Bot.botMove(r.s));
  A(ns && ns.board[r.c] === null, 'greedy captures an enemy trooper in atari');
  r = ataris(); ns = H.apply(r.s, Bot.mctsMove(r.s, FAST ? 40 : 120, 8, 36));
  A(ns && ns.board[r.c] === null, 'mcts captures an enemy trooper in atari');
}

// 2) DEFENSE — my own trooper in atari, with an extending move that saves it,
//    must not be abandoned. After the bot's move the trooper survives with >1 lib.
{
  const s = blank(); const c = idx(4, 4);
  s.board[c] = { c: 0, k: 'T' }; s.placedPly[c] = -10;
  s.board[idx(3, 4)] = { c: 1, k: 'D' }; s.board[idx(4, 3)] = { c: 1, k: 'D' }; s.board[idx(4, 5)] = { c: 1, k: 'D' };
  // only liberty is (5,4); extending there connects to open space (>=2 libs)
  const ns = H.apply(s, Bot.botMove(s));
  const alive = ns && ns.board[c] && ns.board[c].c === 0;
  const safe = alive && H.groupLib(N, ns.board, c).libs >= 2;
  A(safe, 'greedy defends its own trooper out of atari');
}

// 3) HEALTH — self-play is aggressive: games terminate, some end by decapitation,
//    and captures actually happen (the old passive bot rarely captured at all).
{
  // Aggression only shows at real search depth, so the strong thresholds gate the
  // full run; FAST (shallow, few games) just confirms captures happen at all.
  const games = FAST ? 4 : 8, sims = FAST ? 20 : 50;
  let term = 0, decap = 0, caps = 0;
  for (let k = 0; k < games; k++) {
    let s = H.newState(N, k % 2 ? 'garrison' : 'beachhead'), guard = 0;
    while (!s.over && guard++ < 1200) {
      const ns = H.apply(s, Bot.mctsMove(s, sims, 8, 36)); s = ns || H.apply(s, { type: 'pass' });
    }
    if (s.over && [0, 1, -1].includes(s.winner)) term++;
    if (s.winner !== -1 && s.lostTroopers[1 - s.winner] >= s.captureToWin) decap++;
    caps += s.lostTroopers[0] + s.lostTroopers[1];
  }
  A(term === games, `all ${games} self-play games terminate (got ${term})`);
  if (FAST) {
    A(caps >= 1, `captures occur in self-play (got ${caps} total)`);
  } else {
    A(decap >= 1, `at least one game ends by decapitation (got ${decap}/${games})`);
    A(caps / games >= 1, `avg captures/game >= 1 (got ${(caps / games).toFixed(2)})`);
  }
  console.log(`  self-play: ${decap}/${games} decap finishes, ${(caps / games).toFixed(2)} captures/game`);
}

console.log(`\n${ok} checks passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
