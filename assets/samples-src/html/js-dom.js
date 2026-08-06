// Interactive DOM demo for /samples/html.
(function () {
  var list = document.getElementById("list");
  var input = document.getElementById("item");
  var addBtn = document.getElementById("add");
  var clearBtn = document.getElementById("clear");
  var hint = document.getElementById("hint");
  if (!list || !addBtn) {
    console.log("js-dom: missing elements");
    return;
  }

  function addItem(text) {
    var li = document.createElement("li");
    li.textContent = text;
    li.addEventListener("click", function () {
      if (li.classList.contains("done")) {
        li.classList.remove("done");
      } else {
        li.classList.add("done");
      }
      console.log("toggle", li.textContent, li.className);
    });
    list.appendChild(li);
  }

  addBtn.addEventListener("click", function () {
    var t = (input && input.value) ? input.value : "item";
    addItem(t);
    if (input) input.value = "";
    if (hint) hint.textContent = list.childElementCount
      ? list.childElementCount + " item(s)"
      : "list updated";
    console.log("added", t);
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      while (list.firstChild) {
        list.removeChild(list.firstChild);
      }
      if (hint) hint.textContent = "cleared";
      console.log("cleared");
    });
  }

  // Seed one row so the page is useful before any click.
  addItem("seed — click to toggle");
  console.log("js-dom ready");
})();
