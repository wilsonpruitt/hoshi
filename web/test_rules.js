// Node test for the HTML game's rules engine.
//   node web/test_rules.js
// Loads the shared rules module (hoshi-rules.js — same file the page and the
// future Edge Function import) and checks: range tuning, setup placement, group
// capture, suicide-illegality, and that random full games terminate with a valid
// winner across sizes and setups.
const H = require('./hoshi-rules.js');

let ok = 0, fail = 0;
const A = (c, m) => { if (c) ok++; else { fail++; console.log('  FAIL:', m); } };

// 1) drone range tuned to board/range ~2.3
A(H.makeConfig(9).R === 4, 'R(9) should be 4, got ' + H.makeConfig(9).R);
A(H.makeConfig(13).R === 5, 'R(13) should be 5, got ' + H.makeConfig(13).R);

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

// 6) encircle rule: troopers immune to group-liberty capture, die only when
//    every on-board neighbor is an enemy stone.
{
  const N = 9, c = H.idx(N, 4, 4);
  // (a) trooper + friendly drone, group reduced to one liberty; enemy fills it.
  //     Under encircle the drone dies but the trooper survives.
  const s = H.newState(N, 'beachhead'); s.encircle = true;
  const dr = H.idx(N, 4, 5);
  s.board[c] = { c: 1, k: 'T' }; s.placedPly[c] = -10;
  s.board[dr] = { c: 1, k: 'D' };
  for (const p of [[3,4],[5,4],[4,3],[3,5],[5,5]]) s.board[H.idx(N,p[0],p[1])] = { c: 0, k: 'D' };
  s.toMove = 0; s.reserve = [5, 5];
  const ns = H.apply(s, { type: 'trooper', i: H.idx(N, 4, 6) }); // fill last group liberty
  A(ns && ns.board[c] && ns.board[c].k === 'T', 'encircle: trooper survives group-liberty fill');
  A(ns && ns.board[dr] === null, 'encircle: drone in the dead group is removed');
  A(ns && ns.lostTroopers[1] === 0, 'encircle: no trooper counted as captured');

  // (b) lone trooper walled in on all four sides is removed.
  const s2 = H.newState(N, 'beachhead'); s2.encircle = true;
  s2.board[c] = { c: 1, k: 'T' }; s2.placedPly[c] = -10;
  for (const p of [[3,4],[5,4],[4,3]]) s2.board[H.idx(N,p[0],p[1])] = { c: 0, k: 'D' };
  s2.toMove = 0; s2.reserve = [5, 5];
  const ns2 = H.apply(s2, { type: 'trooper', i: H.idx(N, 4, 5) });
  A(ns2 && ns2.board[c] === null, 'encircle: walled-in trooper is captured');
  A(ns2 && ns2.lostTroopers[1] === 1, 'encircle: walled-in capture is counted');

  // (c) same fill as (a) with encircle OFF still removes the whole group (Go).
  const s3 = H.newState(N, 'beachhead'); // encircle defaults false
  s3.board[c] = { c: 1, k: 'T' }; s3.placedPly[c] = -10;
  s3.board[dr] = { c: 1, k: 'D' };
  for (const p of [[3,4],[5,4],[4,3],[3,5],[5,5]]) s3.board[H.idx(N,p[0],p[1])] = { c: 0, k: 'D' };
  s3.toMove = 0; s3.reserve = [5, 5];
  const ns3 = H.apply(s3, { type: 'trooper', i: H.idx(N, 4, 6) });
  A(ns3 && ns3.board[c] === null, 'encircle off: classic group capture removes the trooper');
}

// 7) mobile drones (step mode): relocate within a ready trooper's shadow.
{
  const N = 9, T = H.idx(N, 4, 4), D = H.idx(N, 4, 5);
  const base = () => { const s = H.newState(N, 'beachhead'); s.mobileDrones = true; s.moveMode = 'step';
    s.board[T] = { c: 0, k: 'T' }; s.placedPly[T] = -10; s.board[D] = { c: 0, k: 'D' };
    s.toMove = 0; s.reserve = [5, 5]; return s; };

  // (a) an orthogonal step into the shadow is legal; diagonal / two-cell jumps are not.
  const s = base();
  const legal = H.legalMoves(s).filter(m => m.type === 'move');
  const has = (f, t) => legal.some(m => m.from === f && m.to === t);
  A(has(D, H.idx(N, 4, 6)), 'step: orthogonal move into shadow is legal');
  A(!has(D, H.idx(N, 3, 6)), 'step: diagonal is not a step');
  A(!has(D, H.idx(N, 4, 7)), 'step: a two-cell jump is not a step');
  const ns = H.apply(s, { type: 'move', from: D, to: H.idx(N, 4, 6) });
  A(ns && ns.board[D] === null && ns.board[H.idx(N, 4, 6)] &&
    ns.board[H.idx(N, 4, 6)].k === 'D', 'step: drone relocates one cell');

  // (b) mobile OFF -> no move actions at all.
  const s2 = base(); s2.mobileDrones = false;
  A(H.legalMoves(s2).every(m => m.type !== 'move'), 'mobile off: no move actions');

  // (c) a move can capture: enemy drone in atari, relocate onto its last liberty.
  const s3 = H.newState(N, 'beachhead'); s3.mobileDrones = true; s3.moveMode = 'step';
  const E = H.idx(N, 2, 2);
  s3.board[E] = { c: 1, k: 'D' };
  for (const p of [[1,2],[3,2],[2,1]]) s3.board[H.idx(N,p[0],p[1])] = { c: 0, k: 'D' };
  s3.board[H.idx(N, 2, 4)] = { c: 0, k: 'D' };            // mover's drone, steps left to (2,3)
  s3.board[H.idx(N, 0, 3)] = { c: 0, k: 'T' }; s3.placedPly[H.idx(N, 0, 3)] = -10; // shadows (2,3)
  s3.toMove = 0; s3.reserve = [5, 5];
  const ns3 = H.apply(s3, { type: 'move', from: H.idx(N, 2, 4), to: H.idx(N, 2, 3) });
  A(ns3 && ns3.board[E] === null, 'mobile move onto the last liberty captures');
}

// 8) two-pass rule: one pass doesn't end the game; two in a row do (so you can't
//    end it alone on move 2 and win on near-empty territory).
{
  const N = 9, s = H.newState(N, 'garrison'); s.toMove = 0; s.reserve = [5, 5];
  const a = H.apply(s, { type: 'pass' });
  A(a && !a.over, 'one pass does not end the game');
  const b = H.apply(a, { type: 'pass' });
  A(b && b.over, 'two passes in a row end and score');
  // a real move between passes resets the streak
  const c = H.apply(s, { type: 'pass' });
  const d = H.apply(c, { type: 'trooper', i: H.idx(N, 4, 4) });
  A(d && !d.over, 'a move after a pass keeps the game going');
  const e = H.apply(d, { type: 'pass' });
  A(e && !e.over, 'a lone pass after a move is only the first pass again');
}

console.log(`\n${ok} checks passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
