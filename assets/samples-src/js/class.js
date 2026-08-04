// /samples/js/class.js — ES6 class + method (just engine)
// Run:  /js /samples/js/class.js

class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  mag2() {
    return this.x * this.x + this.y * this.y;
  }
  add(other) {
    return new Point(this.x + other.x, this.y + other.y);
  }
}

var a = new Point(3, 4);
var b = new Point(1, 2);
var c = a.add(b);
console.log("a.mag2() =", a.mag2());
console.log("c = (" + c.x + ", " + c.y + ")");
console.log("c.mag2() =", c.mag2());
return c.mag2();
