// Canvas 2D demo for /samples/html.
(function () {
  var c = document.getElementById("c");
  var status = document.getElementById("status");
  var paintBtn = document.getElementById("paint");
  var clearBtn = document.getElementById("clear");
  if (!c || !c.getContext) {
    if (status) status.textContent = "no canvas / getContext";
    return;
  }
  var ctx = c.getContext("2d");
  if (!ctx) {
    if (status) status.textContent = "getContext returned null";
    return;
  }

  function paint() {
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, c.width, c.height);

    ctx.fillStyle = "#cc785c";
    ctx.fillRect(24, 24, 100, 80);

    ctx.fillStyle = "#e8e0d4";
    ctx.fillRect(140, 40, 60, 60);

    ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.moveTo(220, 30);
    ctx.lineTo(290, 150);
    ctx.stroke();

    if (status) status.textContent = "painted " + c.width + "x" + c.height;
    console.log("canvas paint ok");
  }

  function clear() {
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, c.width, c.height);
    if (status) status.textContent = "cleared";
    console.log("canvas clear");
  }

  if (paintBtn) paintBtn.addEventListener("click", paint);
  if (clearBtn) clearBtn.addEventListener("click", clear);
  paint();
})();
