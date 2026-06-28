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
// Shape: a deep principal-variation spine (best move each side, rank-weighted so
// it survives the probability cutoff to MAX_DEPTH) with shallow shoulders (the
// plausible top-K replies). Off-book positions fall back to live MCTS.
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
const EPS       = +(process.env.EPS       || 0.008); // stop expanding a line below this reach-probability
const WIDTH     = +(process.env.WIDTH     || 8);     // MCTS shortlist width
const BOARD     = +(process.env.BOARD     || 9);
const OUT       = process.env.OUT || path.join(__dirname, 'opening-book.json');
// fog is excluded: its setup is RNG-scattered per game, so no fixed opening exists.
const VARIANTS  = (process.env.VARIANTS || 'beachhead,vanguard,garrison,bastion').split(',');
const depthCap  = BOARD >= 13 ? 28 : 36;
// rank-based weights: principal line keeps ~0.55 so 0.55^d stays above EPS deep;
// shoulders decay fast. Padded/renormalized to the actual candidate count.
const RANK_W = [0.55, 0.25, 0.12, 0.08];

function encode(m){
  if (m.type === 'trooper') return 't' + m.i;
  if (m.type === 'drone')   return 'd' + m.i;
  return 'p'; // pass (mobile off → no 'move')
}

// the plausible continuations to branch into. RANK 0 IS THE STORED BEST MOVE so
// the principal line follows the book's own recommendation deep (best-vs-best);
// the rest are the bot's eval-ranked shoulders (likely opponent deviations).
function continuations(s, best, k){
  const me = s.toMove, scored = [];
  for (const m of H.legalMoves(s)){
    if (m.type === 'pass') continue;            // never branch the book through a pass
    const ns = H.apply(s, m); if (ns === null) continue;
    scored.push([Bot.evaluate(ns, me), m]);
  }
  scored.sort((a, b) => b[0] - a[0]);           // descending: best first
  const bc = encode(best);
  const out = (best.type === 'pass') ? [] : [best];
  for (const [, m] of scored){
    if (out.length >= k) break;
    if (encode(m) !== bc) out.push(m);          // skip the dup of best
  }
  return out;
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
  const cands = continuations(s, best, K);
  if (!cands.length) return;
  const wsum = cands.reduce((a, _c, i) => a + (RANK_W[i] || 0.04), 0);
  cands.forEach((m, i) => {
    const share = (RANK_W[i] || 0.04) / wsum;
    expand(H.apply(s, m), prob * share, depth + 1);
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
