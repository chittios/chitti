# `/samples/js` — scripts for the in-kernel `/js` engine

These files are **authored in-tree** (`assets/samples-src/js/`) and copied into
the gitignored `assets/samples/js/` by `cargo xtask sample-files`. With
`CHITTI_SAMPLE_FILES=1` they are embedded and seeded at boot under
`/samples/js/`.

Run any of them with the Node-style CLI:

```text
/js /samples/js/hello.js
/js /samples/js/hello.js world
/js /samples/js/argv.js a b c
/js /samples/js/fib.js 12
/js /samples/js/math.js
/js /samples/js/class.js
/js /samples/js/json.js
```

Or open one in the editor: `/open /samples/js/fib.js`.

`process.argv` / bare `argv` are set like Node: `argv[0]` = `"js"`,
`argv[1]` = the script path, then user args. Top-level `return` is the
program result (`js=` line).
