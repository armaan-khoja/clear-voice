// Render chosen frames as PNGs for checking: CHROME=/path/to/chrome-headless-shell node render-stills.mjs 300 900 ...
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'node:path';

const frames = process.argv.slice(2).map(Number);
const browserExecutable = process.env.CHROME || null;
const serveUrl = process.env.BUNDLE ? path.resolve(process.env.BUNDLE) : await bundle({entryPoint: path.resolve('src/index.ts')});
console.log('using', serveUrl);
const composition = await selectComposition({serveUrl, id: process.env.COMP || 'Exam', browserExecutable, logLevel: 'verbose'});
console.log('composition', composition.durationInFrames, 'frames');
for (const frame of frames) {
	await renderStill({composition, serveUrl, output: `out/${process.env.COMP || 'Exam'}-${frame}.png`, frame, browserExecutable, logLevel: 'info'});
	console.log('rendered', frame);
}
console.log('ALL-STILLS-DONE');
