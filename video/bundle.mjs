// Bundle the video once into out/bundle (rendering then reuses it): node bundle.mjs
import {bundle} from '@remotion/bundler';
import path from 'node:path';
const url = await bundle({entryPoint: path.resolve('src/index.ts'), outDir: path.resolve('out/bundle'), enableCaching: false});
console.log('bundled', url);
