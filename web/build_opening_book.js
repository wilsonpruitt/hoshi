// HOSHI — deep opening-book generator.
//
// Builds an opening book for the Hard (MCTS) tier so its earliest moves — where
// the board is near-empty, branching is huge, and MCTS is slowest yet least
// decisive — are instant lookups instead of a live search.
//
// Generated WITH THE JS ENGINE THE BROWSER RUNS (hoshi-rules.js + the bot-kernel
// pulled from hoshi.html, exactly like test_bot.js). That guarantees every book
// move is legal and identical to what the live Hard bot would accept — no
// Python↔JS parity risk. Keyed by Hoshi.posKey (now exported), so the browser
// looks up with the identical function.
//
// Shape: TWO deep spines kept above the probability cutoff to MAX_DEPTH — the
// best-vs-best line AND the opponent's greedy/aggressive line (so the book has a
// prepared answer when Medium-style attackers deviate from best-vs-best) — plus
// shallow eval shoulders (other plausible top-K replies). Off-book positions fall
// back to live MCTS.
//
// Usage:
//   node web/build_opening_book.js                 # full run (all det. variants)
//   SIMS=150 MAX_DEPTH=4 VARIANTS=beachhead node web/build_opening_book.js   # quick validate
//
// Knobs (env): SIMS, MAX_DEPTH, K, EPS, WIDTH, BOARD, VARIANTS, OUT
const fs = require('fs');
const path = require('path');
const H = require('./hoshi-rules.js');

// pull the DOM-free bot kernel out of hoshi.html and bind it to the engine
const html = fs.readFileSync(path.join(__dirname, 'hoshi.html'), 'utf8');
const open = html.indexOf('<script id="bot-kernel"');
const kStart = html.indexOf('>', open) + 1;
const kEnd = html.indexOf('</script>', kStart);
const mod = { exports: {} };
new Function('Hoshi', 'module', 'exports', html.slice(kStart, kEnd))(H, mod, mod.exports);
const Bot = mod.exports;

// ── config ──────────────────────────────────────────────────────────────────
const SIMS      = +(process.env.SIMS      || 400);   // MCTS sims per stored answer (offline → can exceed live 240)
const MAX_DEPTH = +(process.env.MAX_DEPTH || 10);    // plies of book depth
const K         = +(process.env.K         || 4);     // continuations expanded per node
const EPS       = +(process.env.EPS       || 0.0015);// stop expanding a line below this reach-probability
                                                     // (low enough that the greedy spine, W_GREEDY share,
                                                     //  survives ~7 plies — see continuations() below)
const WIDTH     = +(process.env.WIDTH     || 8);     // MCTS shortlist width
const BOARD     = +(process.env.BOARD     || 9);
const OUT       = process.env.OUT || path.join(__dirname, 'opening-book.json');
// fog is excluded: its setup is RNG-scattered per game, so no fixed opening exists.
const VARIANTS  = (process.env.VARIANTS || 'beachhead,vanguard,garrison,bastion').split(',');
const depthCap  = BOARD >= 13 ? 28 : 36;
// Continuation weights. We keep TWO lines near-principal so each survives the EPS
// gate to depth: the MCTS best move (our own book line) AND the opponent's GREEDY
// move — the aggressive reply that Medium and human attackers actually play, and
// the line the old book never went deep on (so Hard fell off-book by ply 4-10 and
// played the early game from behind). Eval shoulders decay fast for breadth near
// the top. Renormalized to the actual candidate set in expand().
const W_BEST = 0.50, W_GREEDY = 0.40, W_SHOULDER = 0.05;

function encode(m){
  if (m.type === 'trooper') return 't' + m.i;
  if (m.type === 'drone')   return 'd' + m.i;
  return 'p'; // pass (mobile off → no 'move')
}

// The continuations to branch into, each with its weight. The MCTS best move (our
// book line) and the opponent's greedy move (aggression) are BOTH kept near-
// principal so their lines reach MAX_DEPTH; the rest are eval-ranked shoulders
// (other plausible deviations) that decay fast. Returns parallel {cands, w} arrays.
function continuations(s, best, greedy, k){
  const me = s.toMove, scored = [];
  for (const m of H.legalMoves(s)){
    if (m.type === 'pass') continue;            // never branch the book through a pass
    const ns = H.apply(s, m); if (ns === null) continue;
    scored.push([Bot.evaluate(ns, me), m]);
  }
  scored.sort((a, b) => b[0] - a[0]);           // descending: best first
  const bc = encode(best), gc = encode(greedy);
  const cands = [], w = [];
  if (best.type !== 'pass'){ cands.push(best); w.push(W_BEST); }
  if (greedy.type !== 'pass' && gc !== bc){ cands.push(greedy); w.push(W_GREEDY); } // aggression spine
  for (const [, m] of scored){
    if (cands.length >= k) break;
    const e = encode(m);
    if (e === bc || e === gc) continue;         // skip dups of best / greedy
    cands.push(m); w.push(W_SHOULDER);
  }
  return { cands, w };
}

const book = {};
let nodes = 0, t0 = Date.now();

function expand(s, prob, depth){
  if (s.over || depth >= MAX_DEPTH || prob < EPS) return;
  const key = H.posKey(s.board, s.toMove);
  if (key in book) return;                       // transposition — already answered
  const best = Bot.mctsMove(s, SIMS, WIDTH, depthCap);
  book[key] = encode(best);
  nodes++;
  if (nodes % 50 === 0) process.stdout.write(`\r  ${nodes} nodes, ${Math.round((Date.now()-t0)/1000)}s`);
  const greedy = Bot.botMove(s, 0);             // the aggressive 1-ply reply to also prepare deep for
  const { cands, w } = continuations(s, best, greedy, K);
  if (!cands.length) return;
  const wsum = w.reduce((a, x) => a + x, 0);
  cands.forEach((m, i) => {
    expand(H.apply(s, m), prob * (w[i] / wsum), depth + 1);
  });
}

console.log(`Building book: board ${BOARD}, depth ${MAX_DEPTH}, K ${K}, sims ${SIMS}, variants [${VARIANTS}]`);
for (const v of VARIANTS){
  const before = nodes;
  expand(H.newState(BOARD, v), 1.0, 0);
  console.log(`\r  ${v}: +${nodes - before} nodes (${nodes} total)        `);
}

const out = {
  cfg: { N: BOARD, R: H.makeConfig(BOARD).R, encircle: false, mobile: false },
  meta: { generated_plies: MAX_DEPTH, sims: SIMS, k: K, variants: VARIANTS, nodes },
  book,
};
fs.writeFileSync(OUT, JSON.stringify(out));
const kb = (fs.statSync(OUT).size / 1024).toFixed(1);
console.log(`\nWrote ${OUT} — ${nodes} positions, ${kb} KB, ${Math.round((Date.now()-t0)/1000)}s`);

// ── self-check: every stored move must be legal from its position is implied
// (we only stored engine-returned moves), but spot-check a few replay legally.
let checked = 0, bad = 0;
for (const v of VARIANTS){
  let s = H.newState(BOARD, v);
  for (let d = 0; d < 6 && !s.over; d++){
    const code = book[H.posKey(s.board, s.toMove)];
    if (!code) break;
    const i = +code.slice(1);
    const m = code[0] === 't' ? { type:'trooper', i } : code[0] === 'd' ? { type:'drone', i } : { type:'pass' };
    const ns = H.apply(s, m);
    checked++; if (ns === null) { bad++; console.log(`  BAD book move ${code} for ${v} at ply ${d}`); break; }
    s = ns;
  }
}
console.log(`Self-check: replayed ${checked} principal-line book moves, ${bad} illegal.`);
