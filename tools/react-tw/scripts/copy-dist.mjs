#!/usr/bin/env node
/**
 * Copy Vite dist → assets/samples-src/html/ with stable names for SAMPLE_FILES.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const dist = path.join(root, "dist");
const dest = path.resolve(root, "../../assets/samples-src/html");

function mustExist(p) {
  if (!fs.existsSync(p)) {
    throw new Error(`missing ${p} — run vite build first`);
  }
}

mustExist(dist);
fs.mkdirSync(dest, { recursive: true });

const jsSrc = path.join(dist, "react-tw.js");
const cssSrc = path.join(dist, "react-tw.css");
mustExist(jsSrc);
mustExist(cssSrc);
fs.copyFileSync(jsSrc, path.join(dest, "react-tw.js"));
fs.copyFileSync(cssSrc, path.join(dest, "react-tw.css"));

// Extra chunks (unlikely with a single entry, but copy if present).
for (const name of fs.readdirSync(dist)) {
  if (name.startsWith("react-tw-") && name.endsWith(".js")) {
    fs.copyFileSync(path.join(dist, name), path.join(dest, name));
  }
}

// Hand-assemble HTML: CSS in <head>, IIFE script after #root so mount finds it.
// (Vite puts the classic script in <head>, which races an empty #root.)
// Trailing smoke script: reads the DOM back, so `MOUNTED` is independent
// evidence from the component's own `ALL PASS` (which only proves the bundle
// executed). Both are asserted by the `browse_samples` e2e scenario.
const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>React + Tailwind</title>
  <link rel="stylesheet" href="./react-tw.css" />
</head>
<body class="bg-slate-50 text-slate-900 antialiased">
  <div id="root">
    <div class="flex min-h-screen items-center justify-center p-4">
      <p class="font-mono text-sm text-slate-500">loading React bundle…</p>
    </div>
  </div>
  <script src="./react-tw.js"></script>
  <script>
    (function () {
      // Read the DOM back after the bundle ran. `ALL PASS` above is logged from
      // inside the component, so it proves the bundle *executed*; this proves it
      // reached the document — the two failed independently while React was
      // being brought up here.
      var root = document.getElementById("root");
      var h1 = root && root.querySelector && root.querySelector("h1");
      // Prefer a plain tag selector — attribute *= may be missing in-engine.
      var link = document.querySelector("link");
      var href = link && (link.getAttribute ? link.getAttribute("href") : link.href);
      var css = !!(href && String(href).indexOf("react-tw.css") >= 0);
      if (h1 && css) {
        console.log("react-tw MOUNTED " + h1.textContent);
      } else {
        console.log("react-tw NOT MOUNTED h1=" + !!h1 + " css=" + css);
      }
    })();
  </script>
</body>
</html>
`;
fs.writeFileSync(path.join(dest, "react-tw.html"), html);

console.log(`react-tw: copied dist → ${dest}`);
console.log("  react-tw.html, react-tw.js, react-tw.css");
