// /samples/js/math.js — Math + array builtins the /js engine supports
// Run:  /js /samples/js/math.js

var nums = [3, 1, 4, 1, 5, 9, 2, 6];
var sum = nums.reduce(function (a, b) {
  return a + b;
}, 0);
var squares = nums.map(function (x) {
  return x * x;
});
console.log("nums:", nums.join(","));
console.log("sum:", sum);
console.log("squares:", squares.join(","));
console.log("abs(-7) =", Math.abs(-7));
console.log("max =", Math.max(1, 9, 3));
console.log("floor(3.7) =", Math.floor(3.7));
return sum;
