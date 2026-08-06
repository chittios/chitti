// Shared script for /samples/html — exercises page JS under file:///.
(function () {
  var status = document.getElementById("status");
  if (status) {
    status.textContent = "app.js ran (index)";
    console.log("samples/html index: app.js ok");
  }
  var msg = document.getElementById("msg");
  var countEl = document.getElementById("count");
  var btn = document.getElementById("btn");
  if (msg) {
    msg.textContent = "app.js loaded";
    console.log("samples/html js-demo: app.js ok");
  }
  if (btn && countEl) {
    var n = 0;
    btn.addEventListener("click", function () {
      n += 1;
      countEl.textContent = String(n);
      console.log("click", n);
    });
  }
})();
