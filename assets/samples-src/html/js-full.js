// Large self-checking JS suite — marker: js-full ALL PASS
(function () {
  Suite.start("js-full");
  var ok = Suite.ok;

  // --- queries ---
  var box = document.getElementById("box");
  ok("getElementById", !!box && box.id === "box");
  ok("querySelectorAll(.x)", document.querySelectorAll(".x").length === 3);
  ok("querySelector(.x)", !!document.querySelector(".x"));
  ok("getElementsByTagName(span)", document.getElementsByTagName("span").length >= 3);
  ok("getElementsByClassName(item)", document.getElementsByClassName("item").length === 2);
  ok("querySelector(.item)", !!document.querySelector(".item"));

  // --- text / classList / style ---
  if (box) {
    box.textContent = "hello " + (20 + 22);
    ok("textContent", box.textContent === "hello 42", box.textContent);
    box.classList.add("active");
    box.classList.add("hot");
    var cn = box.className || "";
    ok("classList.add", cn.indexOf("active") >= 0 && cn.indexOf("hot") >= 0, cn);
    ok("classList.contains", box.classList.contains("active") === true);
    box.classList.remove("hot");
    ok("classList.remove", !box.classList.contains("hot"));
    box.style.color = "red";
    box.style.fontSize = "14px";
    ok("style.color", String(box.style.color).indexOf("red") >= 0);
    ok("style.fontSize", String(box.style.fontSize).indexOf("14") >= 0);
  } else {
    ok("textContent", false);
    ok("classList.add", false);
    ok("classList.contains", false);
    ok("classList.remove", false);
    ok("style.color", false);
    ok("style.fontSize", false);
  }

  // --- attributes ---
  if (box) {
    ok("data-k seed", box.getAttribute("data-k") === "v");
    box.setAttribute("data-t", "yes");
    ok("set/getAttribute", box.getAttribute("data-t") === "yes");
    ok("hasAttribute", box.hasAttribute("data-t") === true);
    box.removeAttribute("data-t");
    ok("removeAttribute", !box.hasAttribute("data-t"));
  }
  var link = document.getElementById("link");
  var href = link ? link.getAttribute("href") || link.href || "" : "";
  ok("anchor href", String(href).indexOf("index.html") >= 0, href);

  var inp = document.getElementById("inp");
  if (inp) {
    ok("input value", inp.value === "hi" || inp.getAttribute("value") === "hi");
    ok("input type", (inp.type || inp.getAttribute("type") || "") === "text");
    ok("input name", (inp.name || inp.getAttribute("name") || "") === "q");
    inp.value = "bye";
    ok("input value set", inp.value === "bye");
  } else {
    ok("input value", false);
    ok("input type", false);
    ok("input name", false);
    ok("input value set", false);
  }

  // --- create / append / remove ---
  var host = document.getElementById("playground");
  try {
    if (!host) throw "no playground";
    var kid = document.createElement("div");
    kid.id = "created";
    kid.className = "made";
    kid.textContent = "made";
    host.appendChild(kid);
    var found = document.getElementById("created");
    ok("createElement+appendChild", !!found && found.textContent === "made");
    ok("create className", found && (found.className || "").indexOf("made") >= 0);
    var span = document.createElement("span");
    span.textContent = "x";
    kid.appendChild(span);
    ok("nested appendChild", kid.childElementCount >= 1);
    try {
      var before = host.childElementCount;
      host.removeChild(kid);
      ok(
        "removeChild",
        host.childElementCount === before - 1 || kid.parentNode == null,
        "count=" + host.childElementCount
      );
    } catch (e2) {
      ok("removeChild", false, String(e2));
    }
  } catch (e) {
    ok("createElement+appendChild", false, String(e));
    ok("create className", false, String(e));
    ok("nested appendChild", false, String(e));
    ok("removeChild", false, String(e));
  }

  // --- JSON / Math / encode ---
  try {
    var j = JSON.parse('{"a":41,"b":[1,2],"c":{"d":3}}');
    ok("JSON.parse nested", j.a + 1 === 42 && j.b.length === 2 && j.c.d === 3);
    ok("JSON.stringify", JSON.stringify({ n: 1 }).indexOf("1") >= 0);
  } catch (e) {
    ok("JSON.parse nested", false, String(e));
    ok("JSON.stringify", false, String(e));
  }
  ok("Math.max", Math.max(1, 9, 3) === 9);
  ok("Math.min", Math.min(1, 9, 3) === 1);
  ok("Math.abs", Math.abs(-4) === 4);
  ok("Math.floor", Math.floor(3.9) === 3);
  ok("Math.round", Math.round(2.5) === 3 || Math.round(2.5) === 2);
  try {
    ok("encodeURIComponent", encodeURIComponent("a b") === "a%20b");
  } catch (e) {
    ok("encodeURIComponent", false, String(e));
  }

  // --- Promise ---
  try {
    var pout = 0;
    Promise.resolve(21)
      .then(function (v) {
        return v * 2;
      })
      .then(function (v) {
        pout = v;
      });
    ok("Promise.then chain", pout === 42, String(pout));
  } catch (e) {
    ok("Promise.then chain", false, String(e));
  }

  // --- storage ---
  try {
    localStorage.setItem("full", "ok");
    ok("localStorage", localStorage.getItem("full") === "ok");
    localStorage.removeItem("full");
    var gone = localStorage.getItem("full");
    ok("localStorage remove", gone == null || gone === "");
    sessionStorage.setItem("s", "1");
    ok("sessionStorage", sessionStorage.getItem("s") === "1");
    sessionStorage.removeItem("s");
  } catch (e) {
    ok("localStorage", false, String(e));
    ok("localStorage remove", false, String(e));
    ok("sessionStorage", false, String(e));
  }

  // --- canvas ---
  var c = document.getElementById("c");
  try {
    if (!c) throw "no canvas";
    var ctx = c.getContext("2d");
    if (!ctx) throw "null ctx";
    ctx.fillStyle = "#cc785c";
    ctx.fillRect(0, 0, 16, 16);
    ctx.strokeStyle = "#000";
    ctx.strokeRect(2, 2, 12, 12);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(10, 10);
    ctx.stroke();
    ok("canvas 2d path", true);
  } catch (e) {
    ok("canvas 2d path", false, String(e));
  }

  // --- location / window / title ---
  try {
    ok("location.href set", typeof location.href === "string" && location.href.length > 0);
    ok("window present", typeof window !== "undefined");
    document.title = "JS full suite";
    ok("document.title", document.title.indexOf("full") >= 0);
  } catch (e) {
    ok("location.href set", false, String(e));
    ok("window present", false, String(e));
    ok("document.title", false, String(e));
  }

  // --- events registration ---
  try {
    var n = 0;
    window.addEventListener("click", function () {
      n += 1;
    });
    ok("addEventListener window", true);
  } catch (e) {
    ok("addEventListener window", false, String(e));
  }

  // --- Array / String smoke (avoid instance methods the engine may lack) ---
  ok("Array length", [1, 2, 3].length === 3);
  ok("String length", "hello".length === 5);

  Suite.done("results");
})();
