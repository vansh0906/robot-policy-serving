"""Browser test for the viewer (needs Playwright + Chromium; skipped otherwise)."""
import asyncio
import os
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pw = pytest.importorskip("playwright.sync_api")

from rps.server import Server  # noqa: E402


@pytest.fixture()
def server():
    box, ready = {}, threading.Event()

    def run():
        async def main():
            srv = await Server().start(port=0)
            srv.add_sim(delay_ms=150)
            box["srv"], box["loop"] = srv, asyncio.get_running_loop()
            ready.set()
            await asyncio.Event().wait()
        try:
            asyncio.run(main())
        except RuntimeError:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ready.wait(10)
    yield box["srv"]
    box["loop"].call_soon_threadsafe(box["loop"].stop)


def test_viewer_live_controls_and_replay(server):
    try:
        p = pw.sync_playwright().start()
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    except Exception as e:                                   # browser not installed
        pytest.skip(f"chromium not available: {e}")
    errors = []
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"http://127.0.0.1:{server.port}/")
    page.wait_for_selector("#robots tbody tr", timeout=10000)
    assert page.inner_text("#conn") == "live"
    page.select_option("#newModel", "reach-v1")                # blocked by the release gate
    page.click("#launch")
    page.wait_for_function("document.querySelector('#launchErr').textContent.includes('release gate')", timeout=5000)
    page.click("#robots tbody tr:first-child")
    page.fill("#tx", "0.30"), page.fill("#ty", "-0.30"), page.fill("#tz", "0.25")
    page.click("#sendTarget")
    page.wait_for_selector("#episodes tbody tr", timeout=10000)
    page.click("#episodes tbody tr:first-child")
    page.wait_for_function("document.querySelector('#replayInfo').textContent.includes('episode')", timeout=5000)
    errors = [e for e in errors if "403" not in e]           # the blocked-model launch above is expected
    assert not errors, errors
    browser.close()
    p.stop()
