// Build the self-contained digital-twin viewers:
//   node twin/viewer/build.mjs
// bundles src/main.js (three.js included) with esbuild and inlines the bundle
// plus each PoC's scene JSON into out/twin/poc{1,2}_twin.html.
import { build } from 'esbuild';
import { readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..'); // testbed_design/
const outDir = join(root, 'out', 'twin');
const pocs = (process.argv.slice(2).length ? process.argv.slice(2) : ['poc1', 'poc2']);

const res = await build({
  entryPoints: [join(here, 'src', 'main.js')],
  bundle: true,
  format: 'iife',
  minify: true,
  write: false,
  target: ['es2020'],
  loader: { '.css': 'text' },
  legalComments: 'none',
  logLevel: 'warning',
});
const js = res.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');

for (const poc of pocs) {
  const scenePath = join(outDir, `${poc}_scene.json`);
  if (!existsSync(scenePath)) {
    console.error(`missing ${scenePath} (run: python3 -B twin/twin_model.py)`);
    process.exitCode = 1;
    continue;
  }
  const scene = JSON.parse(readFileSync(scenePath, 'utf8'));
  const data = JSON.stringify(scene).replace(/</g, '\\u003c');
  const title = `${scene.meta.poc} digital twin (tentative)`;
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="testbed_design/twin/viewer/build.mjs">
<title>${title.replace(/[<&]/g, '')}</title>
</head>
<body>
<noscript>This digital twin needs JavaScript and WebGL.</noscript>
<script type="application/json" id="twin-scene">${data}</script>
<script>${js}</script>
</body>
</html>
`;
  const out = join(outDir, `${poc}_twin.html`);
  writeFileSync(out, html);
  const mb = statSync(out).size / 1e6;
  console.log(`${poc}: ${out} (${mb.toFixed(2)} MB)`);
  if (mb > 15) {
    console.error(`${out} exceeds 15 MB`);
    process.exitCode = 1;
  }
}
