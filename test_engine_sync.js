// Drift guard: the Deno Edge-function engine must stay identical to the canonical
// browser engine. They are ONE engine in two module wrappers — this fails the
// moment they diverge so we never validate moves against stale rules.
//   node test_engine_sync.js
const fs = require('fs');
const path = require('path');

const canonPath = path.join(__dirname, 'web/hoshi-rules.js');
const denoPath = path.join(__dirname, 'supabase/functions/_shared/engine.js');
const canon = fs.readFileSync(canonPath, 'utf8');
const deno = fs.readFileSync(denoPath, 'utf8');

// the only allowed difference is the module trailer (CommonJS -> ESM export)
const expected = canon.replace(/if\(typeof module!==.undefined.\) module\.exports=Hoshi;\s*$/, 'export { Hoshi };\n');

let fail = 0;
if (expected === canon) { console.log('FAIL: canonical module trailer not found in web/hoshi-rules.js'); fail = 1; }
else if (deno.trim() !== expected.trim()) {
  fail = 1;
  console.log('FAIL: supabase/functions/_shared/engine.js has drifted from web/hoshi-rules.js');
  console.log('Regenerate with:');
  console.log(`  node -e 'const fs=require("fs");fs.writeFileSync("supabase/functions/_shared/engine.js",` +
    `fs.readFileSync("web/hoshi-rules.js","utf8").replace(/if\\(typeof module!==.undefined.\\) module\\.exports=Hoshi;\\s*$/,"export { Hoshi };\\n"))'`);
}
if (!fail) console.log('engine sync OK — Deno copy matches canonical engine');
process.exit(fail);
