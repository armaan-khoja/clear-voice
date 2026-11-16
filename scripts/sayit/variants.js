// Say It's corrector in lighter forms, applied to one model's text.
//   full      : as the app does it
//   sentences : only when the model heard 2+ words (single words are left alone)
//   nolist    : without the 215 everyday phrases (word prediction + trees only)
//   both      : sentences only and without the phrase list
const C = require('./sayit_corrector.js');
const fs = require('fs');
const F = C.FAM, bank = F.bank, bankSet = F.bankSet, famBi = F.famBi;
const rows = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(JSON.parse);
function pick(heard, useList) {
  if (useList) { F.bank = bank; F.bankSet = bankSet; F.famBi = famBi; } else { F.bank = []; F.bankSet = {}; F.famBi = {}; }
  if (!heard) return '';
  try { const p = C.famPick(heard, [heard]); if (p && p.text) return p.text; } catch (e) {}
  return heard;
}
const nWords = s => s.split(' ').filter(Boolean).length;
const out = rows.map(r => {
  const a = C.noRepeats(r.a), multi = nWords(a) >= 2;
  return { id: r.id, full: pick(a, true), sentences: multi ? pick(a, true) : a, nolist: pick(a, false), both: multi ? pick(a, false) : a };
});
fs.writeFileSync(process.argv[3], out.map(x => JSON.stringify(x)).join('\n'));
console.log('variants', out.length);
