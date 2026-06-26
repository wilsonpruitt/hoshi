// Node test for the HTML game's rules engine.
//   node web/test_rules.js
// Extracts the DOM-free rules block from hoshi.html and checks: range tuning,
// setup placement, group capture, suicide-illegality, and that random full
// games terminate with a valid winner across sizes and setups.
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'hoshi.html'), 'utf8');
// extract a <script id="..."> block by its id (tolerates attributes on the tag)
function scriptById(id) {
  const open = html.indexOf(`<script id="${id}"`);
  const start = html.indexOf('>', open) + 1;
  const end = html.indexOf('</script>', start);
  return html.slice(start, end);
}
const mod = { exports: {} };
new Function('module', 'exports', scriptById('hoshi-rules'))(mod, mod.exports);
const H = mod.exports;

let ok = 0, fail = 0;
const A = (c, m) => { if (c) ok++; else { fail++; console.log('  FAIL:', m); } };

// 1) drone range tuned to board/range ~2.3
A(H.makeConfig(9).R === 4, 'R(9) should be 4, got ' + H.makeConfig(9).R);
A(H.makeConfig(13).R === 6, 'R(13) should be 6, got ' + H.makeConfig(13).R);

// 2) setups place the right number of home troopers
for (const [v, exp] of [['beachhead', 0], ['vanguard', 1], ['garrison', 2], ['bastion', 3], ['fog', 5]]) {
  const s = H.newState(9, v);
  const blue = s.board.filter(b => b && b.c === 0).length;
  A(blue === exp, `${v}: blue home should be ${exp}, got ${blue}`);
  A(s.reserve[0] === 5 - exp, `${v}: blue reserve should be ${5 - exp}, got ${s.reserve[0]}`);
}

// 3) capture removes a surrounded enemy group (group liberties)
{
  const N = 9, s = H.newState(9, 'beachhead'), c = H.idx(N, 4, 4);
  s.board[c] = { c: 1, k: 'D' };
  s.board[H.idx(N, 3, 4)] = { c: 0, k: 'D' };
  s.board[H.idx(N, 5, 4)] = { c: 0, k: 'D' };
  s.board[H.idx(N, 4, 3)] = { c: 0, k: 'D' };
  s.board[H.idx(N, 4, 6)] = { c: 0, k: 'T' }; s.placedPly[H.idx(N, 4, 6)] = -10;
  s.toMove = 0; s.reserve = [5, 5];
  const ns = H.apply(s, { type: 'drone', i: H.idx(N, 4, 5) });
  A(ns !== null, 'capturing drone move should be legal');
  A(ns && ns.board[c] === null, 'surrounded enemy drone should be removed');
}

// 4) suicide illegal unless it captures
{
  const N = 9, s = H.newState(9, 'beachhead');
  s.board[H.idx(N, 3, 4)] = { c: 1, k: 'D' };
  s.board[H.idx(N, 5, 4)] = { c: 1, k: 'D' };
  s.board[H.idx(N, 4, 3)] = { c: 1, k: 'D' };
  s.board[H.idx(N, 4, 5)] = { c: 1, k: 'D' };
  s.toMove = 0; s.reserve = [5, 5];
  const ns = H.apply(s, { type: 'trooper', i: H.idx(N, 4, 4) });
  A(ns === null, 'suicide into a fully enemy-surrounded point should be illegal');
}

// 5) random full games terminate with a valid winner (all sizes & setups)
function randGame(N, v) {
  let s = H.newState(N, v), guard = 0;
  while (!s.over && guard++ < 5000) {
    const ms = H.legalMoves(s);
    const nonpass = ms.filter(m => m.type !== 'pass');
    const pick = (nonpass.length && Math.random() < 0.96) ? nonpass : ms;
    const ns = H.apply(s, pick[Math.floor(Math.random() * pick.length)]);
    if (ns === null) continue;
    s = ns;
  }
  return s;
}
for (const N of [9, 13]) for (const v of ['beachhead', 'garrison', 'bastion', 'fog']) {
  const s = randGame(N, v);
  A(s.over === true, `${N}/${v}: game should terminate`);
  A([0, 1, -1].includes(s.winner), `${N}/${v}: winner should be 0/1/-1, got ${s.winner}`);
}

console.log(`\n${ok} checks passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
