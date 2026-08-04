// /samples/js/fib.js — iterative Fibonacci
// Run:  /js /samples/js/fib.js
//   or: /js /samples/js/fib.js 15

function fib(n) {
  if (n < 2) {
    return n;
  }
  var a = 0;
  var b = 1;
  for (var i = 2; i <= n; i = i + 1) {
    var t = a + b;
    a = b;
    b = t;
  }
  return b;
}

var n = 10;
if (typeof process !== "undefined" && process.argv && process.argv.length > 2) {
  n = Number(process.argv[2]);
  if (typeof n !== "number" || n !== n) {
    n = 10;
  }
}
var result = fib(n);
console.log("fib(" + n + ") =", result);
return result;
