"""Tests 19-20, 28-29: navigation, accessibility script and responsive CSS (static checks)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "app" / "assets" / "portal_a11y.js").read_text(encoding="utf-8")
CSS = (ROOT / "app" / "assets" / "dashboard.css").read_text(encoding="utf-8")
APP = (ROOT / "app" / "app.py").read_text(encoding="utf-8")


def test_19_dashboard_tab_and_default():
    assert '("dashboard", "Dashboard"' in APP and '_pane("dashboard"' in APP
    assert 'DEFAULT_TAB = "dashboard"' in APP


def test_29_a11y_panes_match_navigation():
    panes = re.search(r"var PANES = (\[.*?\]);", JS).group(1)
    assert panes == '["dashboard", "performance", "aemo", "history"]'
    assert "aria-controls" in JS and "aria-labelledby" in JS and '"Home"' in JS and '"End"' in JS


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_29b_keyboard_index_logic():
    script = JS.replace("(function () {", "(function () { var document={getElementById:function(){return null;},addEventListener:function(){},querySelector:function(){return null;},documentElement:{}}; var MutationObserver=function(){this.observe=function(){};}; var window=globalThis;", 1)
    script += "\nvar f=globalThis.__prtNextIndex; console.log([f('ArrowRight',3,4),f('ArrowLeft',0,4),f('Home',2,4),f('End',0,4)].join(','));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "0,3,0,3", out.stderr


def test_28_responsive_css_for_new_components():
    for sel in (".lead-kpis--trio", ".region-strip", ".mc-grid", ".journey", ".region-picker-panel.is-open"):
        assert sel in CSS
    tail = CSS[CSS.index("Portal v3.0.0"):]
    assert "@media (max-width: 900px)" in tail and "@media (max-width: 680px)" in tail
    assert "prefers-reduced-motion" in tail and "min-height: 44px" in tail
    assert ".table-scroll { overflow-x: auto" in CSS


def test_main_block_is_last_so_all_callbacks_register_before_run():
    idx = APP.index('if __name__ == "__main__":')
    assert "@callback(" not in APP[idx:] and "clientside_callback(" not in APP[idx:]
