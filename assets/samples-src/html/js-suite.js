// Self-checking JS suite for /samples/html — runs under /browse file:///.
// Marker: js-suite-v3 (bumped when diagnosing e2e failures).
(function () {
  var results = document.getElementById("results");
  var passed = 0;
  var failed = 0;
  var lines = [];
  var failNames = [];

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

  console.log("js-suite-v3 start");

  // --- DOM queries ---
  var box = document.getElementById("box");
  ok("getElementById", !!box && box.id === "box");

  var xs = document.querySelectorAll(".x");
  ok("querySelectorAll(.x)", xs && xs.length === 2, "len=" + (xs ? xs.length : -1));

  var firstX = document.querySelector(".x");
  ok("querySelector(.x)", !!firstX);

  var byTag = document.getElementsByTagName("span");
  ok("getElementsByTagName(span)", byTag && byTag.length >= 2);

  // --- text / classList / style ---
  if (box) {
    box.textContent = "hello " + (1 + 1);
    ok("textContent set", box.textContent === "hello 2", box.textContent);

    box.classList.add("active");
    var cn = box.className || "";
    ok("classList.add", cn.indexOf("active") >= 0 && cn.indexOf("a") >= 0, cn);

    box.style.color = "red";
    ok("style.color set", (box.style.color || "").indexOf("red") >= 0, String(box.style.color));
  } else {
    ok("textContent set", false, "no #box");
    ok("classList.add", false, "no #box");
    ok("style.color set", false, "no #box");
  }

  // --- attributes (data-* + typed href) ---
  var link = document.getElementById("link");
  if (box) {
    var dk0 = box.getAttribute("data-k");
    console.log("diag data-k=[" + dk0 + "] has=" + box.hasAttribute("data-k"));
    box.setAttribute("data-t", "yes");
    ok("setAttribute/getAttribute", box.getAttribute("data-t") === "yes");
    ok("hasAttribute", box.hasAttribute("data-t") === true);
    ok("data-k seed attr", dk0 === "v", "got=[" + dk0 + "]");
  } else {
    ok("setAttribute/getAttribute", false);
    ok("hasAttribute", false);
    ok("data-k seed attr", false);
  }

  var hrefAttr = link ? link.getAttribute("href") : null;
  var hrefProp = link ? link.href : null;
  console.log("diag href attr=[" + hrefAttr + "] prop=[" + hrefProp + "]");
  var hrefOk = false;
  if (link) {
    var h = hrefAttr || hrefProp || "";
    hrefOk = String(h).indexOf("index.html") >= 0;
  }
  ok("anchor href", hrefOk, "attr=[" + hrefAttr + "] prop=[" + hrefProp + "]");

  // --- createElement / appendChild ---
  var host = document.getElementById("playground");
  try {
    if (!host) throw "no #playground";
    var kid = document.createElement("div");
    kid.id = "created";
    kid.textContent = "made";
    host.appendChild(kid);
    var found = document.getElementById("created");
    ok("createElement+appendChild", !!found && found.textContent === "made");
  } catch (e) {
    ok("createElement+appendChild", false, String(e));
  }

  // --- JSON ---
  try {
    var j = JSON.parse('{"a":41,"b":[1,2]}');
    ok("JSON.parse", j.a + 1 === 42 && j.b.length === 2);
    var js = JSON.stringify({ n: 1 });
    ok("JSON.stringify", typeof js === "string" && js.indexOf("1") >= 0, js);
  } catch (e) {
    ok("JSON.parse", false, String(e));
    ok("JSON.stringify", false, String(e));
  }

  // --- localStorage ---
  try {
    localStorage.setItem("suite", "ok");
    ok("localStorage set/get", localStorage.getItem("suite") === "ok");
    localStorage.removeItem("suite");
    var gone = localStorage.getItem("suite");
    ok("localStorage remove", gone == null || gone === "");
  } catch (e) {
    ok("localStorage set/get", false, String(e));
    ok("localStorage remove", false, String(e));
  }

  // --- canvas ---
  var c = document.getElementById("c");
  try {
    if (!c) throw "no #c";
    var ctx = c.getContext("2d");
    if (!ctx) throw "null context";
    ctx.fillStyle = "red";
    ctx.fillRect(0, 0, 10, 10);
    ok("canvas getContext+fillRect", true);
  } catch (e) {
    ok("canvas getContext+fillRect", false, String(e));
  }

  ok("console.log reachable", true);

  // --- live click handler ---
  var liveBtn = document.getElementById("live-btn");
  var liveOut = document.getElementById("live-out");
  var clicks = 0;
  if (liveBtn && liveOut) {
    liveBtn.addEventListener("click", function () {
      clicks += 1;
      liveOut.textContent = String(clicks);
      console.log("live click", clicks);
    });
    ok("addEventListener registered", true);
  } else {
    ok("addEventListener registered", false);
  }

  // One-line summary for e2e (always printed).
  if (failed === 0) {
    console.log("js-suite ALL PASS (" + passed + ")");
  } else {
    console.log(
      "js-suite FAIL " + failed + " of " + (passed + failed) + " :: " + failNames.join(",")
    );
  }

  // Render report via DOM nodes (innerHTML is text-only in this engine).
  if (results) {
    try {
      while (results.firstChild) {
        results.removeChild(results.firstChild);
      }
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
        (failed === 0 ? " — all good" : " — check FAIL lines");
      results.appendChild(sum);
    } catch (e) {
      console.log("js-suite render err " + e);
    }
  }
})();
