// Apply Say It's corrector, exactly as the app's finish() does, to saved transcripts.
// in: JSON lines {id, a, b}  (a = primary model's text, b = second model's text or null)
const C = require('./sayit_corrector.js');
const fs = require('fs');
const rows = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(JSON.parse);
const pick = (heard, extra) => { if (!heard) return ''; try { const p = C.famPick(heard, extra); if (p && p.text) return p.text; } catch (e) {} return heard; };
const out = rows.map(r => {
  const a = C.noRepeats(r.a), b = r.b == null ? null : C.noRepeats(r.b);
  return { id: r.id, a_norep: a, a_fix: pick(a, [a]), b_fix: b == null ? null : pick(b, [b]), both: b == null ? null : pick(a, [a, b]) };
});
fs.writeFileSync(process.argv[3], out.map(x => JSON.stringify(x)).join('\n'));
console.log('corrected', out.length);
