# React + Tailwind sample for `/browse file:///samples/html/react-tw.html`

Offline page built with Vite, React 18, and Tailwind 3. The production bundle
is copied into `assets/samples-src/html/` (checked in) so
`CHITTI_SAMPLE_FILES` embeds it without needing npm on every build.

```sh
cd tools/react-tw
npm install
npm run build   # vite build + copy into assets/samples-src/html/
```

Then refresh the gitignored corpus copy if you boot with samples:

```sh
cargo xtask sample-files --refresh
```

Two independent markers, both asserted by the `browse_samples` e2e scenario:

- `react-tw ALL PASS` — logged from *inside* the mounted component, so it proves
  the React production bundle executed on the in-OS engine.
- `react-tw MOUNTED <h1 text>` — logged by a trailing inline script that reads
  the document back, so it proves the commit reached the DOM.

They failed independently during bring-up (the bundle rendered and then threw in
commit), which is why there are two.
