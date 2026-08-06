// Shared PASS/FAIL harness for /samples/html JS suites.
// Exposes: Suite.start(name), Suite.ok(name, cond, detail), Suite.done(resultsElId)
(function (g) {
  var passed = 0;
  var failed = 0;
  var lines = [];
  var failNames = [];
  var suiteName = "suite";

  function ok(name, cond, detail) {
    if (cond) {
      passed += 1;
      lines.push({ ok: true, text: "PASS  " + name });
    } else {
      failed += 1;
      failNames.push(name + (detail ? "(" + detail + ")" : ""));
      lines.push({
        ok: false,
        text: "FAIL  " + name + (detail ? " — " + detail : ""),
      });
    }
  }

  function start(name) {
    suiteName = name || "suite";
    passed = 0;
    failed = 0;
    lines = [];
    failNames = [];
    console.log(suiteName + " start");
  }

  function done(resultsId) {
    if (failed === 0) {
      console.log(suiteName + " ALL PASS (" + passed + ")");
    } else {
      console.log(
        suiteName +
          " FAIL " +
          failed +
          " of " +
          (passed + failed) +
          " :: " +
          failNames.join(",")
      );
    }
    var results = resultsId ? document.getElementById(resultsId) : null;
    if (!results) return;
    try {
      while (results.firstChild) results.removeChild(results.firstChild);
      for (var i = 0; i < lines.length; i++) {
        var row = document.createElement("div");
        row.className = lines[i].ok ? "pass" : "fail";
        row.textContent = lines[i].text;
        results.appendChild(row);
      }
      var sum = document.createElement("div");
      sum.className = "sum";
      sum.textContent =
        passed +
        " passed, " +
        failed +
        " failed" +
        (failed === 0 ? " — all good" : "");
      results.appendChild(sum);
    } catch (e) {
      console.log(suiteName + " render err " + e);
    }
  }

  g.Suite = { start: start, ok: ok, done: done };
})(typeof window !== "undefined" ? window : this);
