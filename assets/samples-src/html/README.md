# /samples/html

Local HTML pages for the in-OS browser (`/browse`).

```text
/browse file:///samples/html/index.html
/browse /samples/html/js-suite.html
/browse /samples/html/js-full.html
/browse /samples/html/js-fetch.html
/browse /samples/html/js-iframe.html
/browse /samples/html/css-full.html
```

## Pages

| File | What it exercises |
|------|-------------------|
| `hello.html` | inline script + basic CSS |
| `css-demo.html` | flex, grid, `@media`, `calc`, `var`, `::before` |
| `css-suite.html` | selectors, float/clear, tables/rowspan, position; `suite.css` `@import`s `theme.css` |
| `css-full.html` | large visual CSS checklist (`full.css`) |
| `css-hn.html` | HN-like table + link colour cascade |
| `js-demo.html` | relative `styles.css` + `app.js` + click counter |
| `js-suite.html` | self-checking DOM / `localStorage` / canvas / events (`js-suite.js`) |
| `js-full.html` | large DOM / Promise / Math / storage / canvas suite |
| `js-fetch.html` | `fetch` relative + `file:///samples/…` JSON (`fetch-data.json`) |
| `js-iframe.html` | iframe `src`/`srcdoc` + same-window `postMessage` |
| `react-tw.html` | Vite + React 18 + Tailwind 3 (`tools/react-tw` → dist) |
| `js-dom.html` | interactive list via `createElement` / `classList` |
| `js-canvas.html` | `getContext('2d')` fill / stroke |

```text
/browse /samples/html/react-tw.html
```

Relative CSS/JS resolve against the document's `file:///` URL and are read
from the Synapse store — no network required. Page `fetch` may read other
files under `/samples/` only (SSRF-closed elsewhere).

On self-checking JS pages, every line should say **PASS**. A **FAIL** means that
binding or subresource path needs a look. E2E looks for `js-suite ALL PASS`,
`js-full ALL PASS`, `js-fetch ALL PASS`, `js-iframe ALL PASS`, and
`react-tw ALL PASS`.
