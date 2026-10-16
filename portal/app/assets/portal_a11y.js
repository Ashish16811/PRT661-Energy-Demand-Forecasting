/* PRT661 portal v3.0.0 — keyboard access for the preserved v2 primary navigation + small UI helpers.
   Dash renders dcc.Tab as plain <div>s; this makes them focusable, announces
   them as tabs (linked to their panels), and supports Enter/Space + arrow keys,
   Home and End. v3 keeps the Dashboard as the default view.
   No dependencies. */
(function () {
  var PANES = ["dashboard", "performance", "aemo", "history"];
  window.__prtPanes = PANES;   // exposed for tests / debugging

  function enhance() {
    var list = document.getElementById("tabs");
    if (!list) return;
    if (list.getAttribute("role") !== "tablist") {
      list.setAttribute("role", "tablist");
      list.setAttribute("aria-label", "Portal views");
    }
    var tabs = list.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
      var t = tabs[i];
      var key = PANES[i];
      var selected = t.classList.contains("tab--selected") || t.classList.contains("nav-tab--active");
      if (!t.id) t.id = "nav-tab-" + (key || i);
      if (t.getAttribute("role") !== "tab") t.setAttribute("role", "tab");
      var ti = selected ? "0" : "-1";
      if (t.getAttribute("tabindex") !== ti) t.setAttribute("tabindex", ti);
      var s = selected ? "true" : "false";
      if (t.getAttribute("aria-selected") !== s) t.setAttribute("aria-selected", s);
      if (key && t.getAttribute("aria-controls") !== "pane-" + key) t.setAttribute("aria-controls", "pane-" + key);
      var pane = key && document.getElementById("pane-" + key);
      if (pane) {
        if (pane.getAttribute("aria-labelledby") !== t.id) pane.setAttribute("aria-labelledby", t.id);
        if (pane.getAttribute("tabindex") !== "-1") pane.setAttribute("tabindex", "-1");
      }
    }
  }

  function nextIndex(key, i, n) {
    if (key === "ArrowRight") return (i + 1) % n;
    if (key === "ArrowLeft") return (i - 1 + n) % n;
    if (key === "Home") return 0;
    if (key === "End") return n - 1;
    return i;
  }
  window.__prtNextIndex = nextIndex;  // pure helper, exercised by tests

  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (!t || !t.classList || !t.classList.contains("tab") || !t.closest("#tabs")) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      t.click();
      return;
    }
    if (e.key === "ArrowRight" || e.key === "ArrowLeft" || e.key === "Home" || e.key === "End") {
      var tabs = Array.prototype.slice.call(t.parentElement.querySelectorAll(".tab"));
      var next = nextIndex(e.key, tabs.indexOf(t), tabs.length);
      e.preventDefault();
      tabs[next].focus();
      tabs[next].scrollIntoView({block: "nearest", inline: "nearest"});
    }
  });

  // Views stay mounted and are shown/hidden with CSS. When a view becomes
  // visible, nudge Plotly so charts rendered while hidden fit their container.
  var activeKey = null;
  function syncActivePane() {
    var active = document.querySelector(".pane.is-active");
    var key = active ? active.id : null;
    if (key && key !== activeKey) {
      activeKey = key;
      window.requestAnimationFrame(function () { window.dispatchEvent(new Event("resize")); });
      var sel = document.querySelector("#tabs .tab.tab--selected");
      if (sel && sel.scrollIntoView) sel.scrollIntoView({block: "nearest", inline: "nearest"});
    }
  }

  var pending = false;
  new MutationObserver(function () {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () { pending = false; enhance(); syncActivePane(); });
  }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
})();
