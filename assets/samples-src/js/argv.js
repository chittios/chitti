// /samples/js/argv.js — Node-shaped process.argv / bare argv
// Run:  /js /samples/js/argv.js one two three

console.log("argv length =", argv.length);
for (var i = 0; i < argv.length; i = i + 1) {
  console.log("  argv[" + i + "] =", argv[i]);
}
// Same array lives on process.argv for Node-ish scripts.
return argv.length;
