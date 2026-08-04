// /samples/js/hello.js — minimal /js script
// Run:  /js /samples/js/hello.js
//   or: /js /samples/js/hello.js world

console.log("hello from ChittiOS /js");
if (typeof process !== "undefined" && process.argv && process.argv.length > 2) {
  console.log("greeting:", process.argv[2]);
}
return 42;
