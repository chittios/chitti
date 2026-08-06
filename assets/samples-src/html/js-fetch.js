// fetch suite — relative + absolute file:///samples/ JSON
(function () {
  Suite.start("js-fetch");
  var ok = Suite.ok;

  ok(
    "fetch is callable",
    typeof fetch === "function" || typeof fetch === "object",
    typeof fetch
  );

  // Relative URL resolves against the document file:/// base.
  var relOk = false;
  var relDetail = "";
  try {
    fetch("fetch-data.json")
      .then(function (r) {
        ok("relative Response.ok", !!r && (r.ok === true || r.status === 200 || typeof r.text === "function"));
        return r.json();
      })
      .then(function (j) {
        relOk = !!(j && j.ok === true && j.n === 42 && j.items && j.items.length === 3);
        relDetail = j ? JSON.stringify(j).slice(0, 80) : "null";
        ok("relative fetch json", relOk, relDetail);
        ok("relative nested.x", j && j.nested && j.nested.x === 1);
      });
  } catch (e) {
    ok("relative Response.ok", false, String(e));
    ok("relative fetch json", false, String(e));
    ok("relative nested.x", false, String(e));
  }

  // Absolute file URL under /samples/ only.
  try {
    fetch("file:///samples/html/fetch-data.json")
      .then(function (r) {
        return r.text();
      })
      .then(function (t) {
        ok("absolute file text", typeof t === "string" && t.indexOf('"suite": "fetch"') >= 0, t.slice(0, 60));
        var j2 = JSON.parse(t);
        ok("absolute parse", j2.suite === "fetch");
      });
  } catch (e) {
    ok("absolute file text", false, String(e));
    ok("absolute parse", false, String(e));
  }

  // HEAD is allowed by policy (may return same body stub).
  try {
    fetch("fetch-data.json", { method: "HEAD" }).then(function (r) {
      ok("HEAD method", !!r);
    });
  } catch (e) {
    ok("HEAD method", false, String(e));
  }

  // POST must fail closed (policy) — error JSON body.
  try {
    fetch("fetch-data.json", { method: "POST", body: "x" })
      .then(function (r) {
        return r.text();
      })
      .then(function (t) {
        ok(
          "POST refused",
          typeof t === "string" && t.indexOf("error") >= 0,
          String(t).slice(0, 80)
        );
      });
  } catch (e) {
    ok("POST refused", true, "threw");
  }

  // Outside /samples/ must fail.
  try {
    fetch("file:///etc/passwd")
      .then(function (r) {
        return r.text();
      })
      .then(function (t) {
        ok(
          "file outside samples refused",
          typeof t === "string" && t.indexOf("error") >= 0,
          t.slice(0, 80)
        );
      });
  } catch (e) {
    ok("file outside samples refused", true, "threw");
  }

  Suite.done("results");
})();
