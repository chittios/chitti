// /samples/js/json.js — JSON.stringify / JSON.parse
// Run:  /js /samples/js/json.js

var obj = { name: "chitti", version: 1, tags: ["os", "agent"] };
var text = JSON.stringify(obj);
console.log("json:", text);
var back = JSON.parse(text);
console.log("name:", back.name);
console.log("tags length:", back.tags.length);
return back.tags.length;
