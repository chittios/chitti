// iframe + postMessage suite
(function () {
  Suite.start("js-iframe");
  var ok = Suite.ok;

  var child = document.getElementById("child");
  var inline = document.getElementById("inline");
  ok("iframe#child present", !!child);
  ok("iframe#inline present", !!inline);

  var src = child ? child.getAttribute("src") || child.src || "" : "";
  ok("iframe src attr", String(src).indexOf("iframe-child.html") >= 0, src);

  var srcdoc = inline ? inline.getAttribute("srcdoc") || "" : "";
  ok("iframe srcdoc attr", srcdoc.indexOf("srcdoc child") >= 0 || srcdoc.indexOf("srcdoc-hi") >= 0, srcdoc.slice(0, 40));

  var frames = document.getElementsByTagName("iframe");
  ok("getElementsByTagName(iframe)", frames && frames.length >= 2, String(frames ? frames.length : -1));

  // Same-window postMessage → message listener (synchronous delivery).
  var got = "";
  var originGot = "";
  try {
    window.addEventListener("message", function (ev) {
      if (ev && ev.data != null) {
        got = String(ev.data);
        if (ev.origin != null) originGot = String(ev.origin);
      }
    });
    ok("addEventListener(message)", true);
  } catch (e) {
    ok("addEventListener(message)", false, String(e));
  }

  try {
    window.postMessage("ping-self", "*");
    ok("postMessage self delivered", got === "ping-self", "got=[" + got + "]");
    ok("MessageEvent.origin string", typeof originGot === "string");
  } catch (e) {
    ok("postMessage self delivered", false, String(e));
    ok("MessageEvent.origin string", false, String(e));
  }

  // Bare postMessage alias
  var got2 = "";
  try {
    window.addEventListener("message", function (ev) {
      if (ev && String(ev.data) === "bare") got2 = "bare";
    });
    postMessage("bare", "*");
    ok("bare postMessage", got2 === "bare", got2);
  } catch (e) {
    ok("bare postMessage", false, String(e));
  }

  // parent.postMessage must not throw (queues outbound toward parent).
  try {
    if (window.parent) {
      window.parent.postMessage("to-parent", "*");
    }
    ok("parent.postMessage no-throw", true);
  } catch (e) {
    ok("parent.postMessage no-throw", false, String(e));
  }

  // Second self message still delivers
  var n = 0;
  try {
    window.addEventListener("message", function (ev) {
      if (ev && String(ev.data).indexOf("count") === 0) n += 1;
    });
    window.postMessage("count-1", "*");
    window.postMessage("count-2", "*");
    ok("multi postMessage", n === 2, "n=" + n);
  } catch (e) {
    ok("multi postMessage", false, String(e));
  }

  Suite.done("results");
})();
