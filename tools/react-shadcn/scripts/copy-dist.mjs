#!/usr/bin/env node
/**
 * Copy the Vite dist → assets/samples-src/html/ with stable names for
 * SAMPLE_FILES, and hand-assemble the page.
 *
 * Same shape as tools/react-tw/scripts/copy-dist.mjs: CSS in <head>, the IIFE
 * after #root (Vite puts the script in <head>, which races an empty container),
 * then a trailing inline script that reads the DOM *back*. `ALL PASS` is logged
 * from inside the mounted app and proves the bundle executed; `MOUNTED` is
 * logged here and proves the commit reached the document. They failed
 * independently while React was being brought up on the in-OS engine, which is
 * why both exist.
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

const jsSrc = path.join(dist, "shadcn.js");
const cssSrc = path.join(dist, "shadcn.css");
mustExist(jsSrc);
mustExist(cssSrc);
fs.copyFileSync(jsSrc, path.join(dest, "shadcn.js"));
fs.copyFileSync(cssSrc, path.join(dest, "shadcn.css"));

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>shadcn/ui gallery</title>
  <link rel="stylesheet" href="./shadcn.css" />
</head>
<body class="bg-background text-foreground">
  <div id="root">
    <div class="flex min-h-screen items-center justify-center p-4">
      <p class="font-mono text-sm text-muted-foreground">loading shadcn/ui bundle…</p>
    </div>
  </div>
  <script src="./shadcn.js"></script>
  <script>
    (function () {
      // Read the document back: every section the gallery renders gets an id,
      // so a component that failed to mount is NAMED instead of silently
      // missing from a screenshot.
      var ids = ["button", "badge", "card", "input", "label", "textarea", "checkbox",
                 "switch", "separator", "progress", "avatar", "skeleton", "alert",
                 "tabs", "accordion", "table"];
      var missing = [];
      for (var i = 0; i < ids.length; i++) {
        if (!document.getElementById("sec-" + ids[i])) { missing.push(ids[i]); }
      }
      var root = document.getElementById("root");
      var h1 = root && root.querySelector && root.querySelector("h1");
      var link = document.querySelector("link");
      var href = link && (link.getAttribute ? link.getAttribute("href") : link.href);
      var css = !!(href && String(href).indexOf("shadcn.css") >= 0);
      // Count what actually reached the document. "the section frames paint but
      // they are empty" and "the components never rendered" look identical on
      // screen, and only these counts tell them apart.
      var counts = "buttons=" + document.querySelectorAll("button").length +
                   " inputs=" + document.querySelectorAll("input").length +
                   " divs=" + document.querySelectorAll("div").length;
      if (h1 && css && missing.length === 0) {
        console.log("shadcn MOUNTED " + h1.textContent + " " + counts);
      } else {
        console.log(
          "shadcn NOT MOUNTED h1=" + !!h1 + " css=" + css + " missing=" + missing.join(",")
        );
      }
    })();
  </script>
</body>
</html>
`;
fs.writeFileSync(path.join(dest, "shadcn.html"), html);

console.log(`shadcn: copied dist → ${dest}`);
console.log("  shadcn.html, shadcn.js, shadcn.css");
