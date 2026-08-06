# shadcn/ui gallery for `/browse file:///samples/html/shadcn.html`

Real [shadcn/ui](https://ui.shadcn.com) components (new-york style, MIT) on
Vite + React 18 + Tailwind 3, built offline and rendered by the in-OS browser.
The production bundle is copied into `assets/samples-src/html/` (checked in) so
`CHITTI_SAMPLE_FILES` embeds it without needing npm on every build.

```sh
cd tools/react-shadcn
npm install
npm run build   # vite build + copy into assets/samples-src/html/
cd ../.. && cargo xtask sample-files --refresh
```

## The 16 components it renders

button · badge · card · input · label · textarea · checkbox · switch ·
separator · progress · avatar · skeleton · alert · tabs · accordion · table

Ten of them are backed by Radix primitives, which is the point: they exercise
`forwardRef`, computed keys, `WeakMap`, `closest`, `createElementNS`,
`getComputedStyle` and ReactDOM's input value-tracking — the surface a real
component library needs and a toy DOM does not have.

## How it reports failure

A gallery that renders a blank page tells you nothing, so this one names its own
faults, in three layers:

1. **`shadcn IMPORTS OK <n>`** / **`shadcn MISSING <names>`** — checked before
   rendering, because React's "element type is invalid" error says `got:
   undefined` and names nothing.
2. **`shadcn SECTION FAIL <id>: <error>`** — every section is wrapped in its own
   error boundary, so one broken component names itself instead of taking the
   page down.
3. **`shadcn MOUNTED <title> buttons=<n> inputs=<n> divs=<n>`** — logged by an
   inline script that reads the document *back* after the bundle ran. The counts
   matter: "the section frames painted but they are empty" and "the components
   never rendered" look identical on screen, and only the counts tell them
   apart.

The `browse_samples` e2e scenario asserts all three, plus the browser's own
`rects=/runs=/ctrls=` layout counts — a commit that stops part-way down the tree
leaves every frame drawn and empty, which the DOM counts alone cannot see.
