"""Screenshot the running viewer, and report any console error it produced.

Why this exists: the viewer is WebGL plus a websocket, so a screenshot of it has
to be taken by a real browser after the page has been fed real frames. ``macOS
screencapture`` needs a screen-recording grant this environment does not have,
and there is no headless-browser package in the venv, so this drives the Chrome
that is already installed over the DevTools protocol using ``websockets`` (a
uvicorn dependency, already present).

It also answers the only question a screenshot cannot: whether the page logged
an error while it ran. Every ``console.error`` / uncaught exception / failed
request is collected over the same window and printed, so "no console errors" is
a measurement rather than an impression.

Usage::

    python -m scripts.viewer_shot --url http://127.0.0.1:8768/ \
        --out outputs/plast2/viewer_plastic_v2.png --wait 45

``--wait`` is real seconds with the page live, so give it enough for the number
of hands you want on screen (the viewer's ``--pace`` sets the rate).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

#: Swiftshader so the point cloud actually rasterises in headless; the rest is
#: the usual "this is a throwaway profile" set.
CHROME_FLAGS: Sequence[str] = (
    "--headless=new",
    "--disable-gpu-sandbox",
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--force-device-scale-factor=1",
)


def _http_json(url: str, timeout: float = 1.0) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def wait_for_devtools(port: int, deadline: float) -> str:
    """Block until Chrome's DevTools endpoint answers; return its websocket URL."""
    last: Optional[Exception] = None
    while time.time() < deadline:
        try:
            info = _http_json(f"http://127.0.0.1:{port}/json/version")
            return str(info["webSocketDebuggerUrl"])  # type: ignore[index]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            last = exc
            time.sleep(0.2)
    raise RuntimeError(f"Chrome DevTools on port {port} never answered: {last}")


async def capture(ws_url: str, page_url: str, out: Path, wait_s: float,
                  width: int, height: int, settle: float) -> Dict[str, object]:
    import websockets  # noqa: PLC0415

    logs: List[dict] = []
    msg_id = 0

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        pending: Dict[int, asyncio.Future] = {}
        session: Dict[str, Optional[str]] = {"id": None}

        async def send(method: str, params: Optional[dict] = None,
                       session_id: Optional[str] = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            payload: dict = {"id": msg_id, "method": method, "params": params or {}}
            if session_id:
                payload["sessionId"] = session_id
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            pending[msg_id] = fut
            await ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=90.0)

        async def pump() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if "id" in msg and msg["id"] in pending:
                    fut = pending.pop(msg["id"])
                    if not fut.done():
                        fut.set_result(msg.get("result", {}) if "error" not in msg
                                       else {"__error__": msg["error"]})
                    continue
                method = msg.get("method", "")
                p = msg.get("params", {})
                if method == "Runtime.consoleAPICalled" and p.get("type") in (
                        "error", "warning", "assert"):
                    logs.append({"kind": "console." + str(p.get("type")),
                                 "text": _args_text(p.get("args", []))})
                elif method == "Runtime.exceptionThrown":
                    det = p.get("exceptionDetails", {})
                    logs.append({"kind": "exception",
                                 "text": str(det.get("text", ""))
                                 + " " + str((det.get("exception") or {}).get("description", ""))})
                elif method == "Log.entryAdded":
                    entry = p.get("entry", {})
                    if entry.get("level") in ("error", "warning"):
                        logs.append({"kind": "log." + str(entry.get("level")),
                                     "text": f"{entry.get('source')}: {entry.get('text')}"})

        pumper = asyncio.create_task(pump())
        try:
            targets = await send("Target.getTargets")
            page = next((t for t in targets.get("targetInfos", [])
                         if t.get("type") == "page"), None)
            if page is None:
                raise RuntimeError("Chrome started with no page target")
            attached = await send("Target.attachToTarget",
                                  {"targetId": page["targetId"], "flatten": True})
            sid = str(attached["sessionId"])
            session["id"] = sid
            await send("Runtime.enable", session_id=sid)
            await send("Log.enable", session_id=sid)
            await send("Page.enable", session_id=sid)
            await send("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1,
                "mobile": False,
            }, session_id=sid)
            await send("Page.navigate", {"url": page_url}, session_id=sid)
            # Pause and speed are server-side and shared by every client, so a
            # previous visitor may have left the loop stopped or slowed. Put it
            # back to a known state before counting on hands arriving.
            await asyncio.sleep(2.0)
            await send("Runtime.evaluate", {
                "expression": "window.flyClient && (window.flyClient.send({cmd:'play'}), "
                              "window.flyClient.send({cmd:'speed', value:1}))",
            }, session_id=sid)
            await asyncio.sleep(max(0.0, wait_s - 2.0))
            if settle > 0:
                # Land the shot on a *finished* hand. The panels dim themselves
                # while a window is running and only fill in when the wave has
                # played back, so a blind capture usually catches the dimmed
                # state. Slowing the loop widens that settled interval, then the
                # page is polled for it; the brain's glow is still decaying from
                # the wave at that point, which is the frame worth having.
                await send("Runtime.evaluate", {
                    "expression": "window.flyClient && window.flyClient.send("
                                  "{cmd:'speed', value:0.25})",
                }, session_id=sid)
                deadline = time.time() + settle
                while time.time() < deadline:
                    probe = await send("Runtime.evaluate", {
                        "expression": _SETTLED_JS, "returnByValue": True,
                    }, session_id=sid)
                    if (probe.get("result") or {}).get("value"):
                        break
                    await asyncio.sleep(0.04)
            shot = await send("Page.captureScreenshot",
                              {"format": "png", "captureBeyondViewport": False},
                              session_id=sid)
            if "__error__" in shot:
                raise RuntimeError(f"captureScreenshot failed: {shot['__error__']}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(shot["data"]))
            # Whatever the page itself thinks it is showing, for the report.
            probe = await send("Runtime.evaluate", {
                "expression": _PROBE_JS, "returnByValue": True, "awaitPromise": False,
            }, session_id=sid)
            value = (probe.get("result") or {}).get("value")
        finally:
            pumper.cancel()
    return {"logs": logs, "probe": value}


def _args_text(args: Sequence[dict]) -> str:
    parts: List[str] = []
    for a in args:
        for key in ("value", "description", "unserializableValue"):
            if key in a:
                parts.append(str(a[key]))
                break
        else:
            parts.append(str(a.get("type", "?")))
    return " ".join(parts)


#: True when the decision panel is showing a finished hand rather than dimming
#: itself through a running window.
_SETTLED_JS = """
(function () {
  // The end-of-run toast sits over the middle of the screen and fades out over
  // 0.2 s, so its *computed opacity* is the condition, not its class: the class
  // drops the moment the fade starts and the ghost is still on screen.
  var toast = document.getElementById('toast');
  if (toast && parseFloat(getComputedStyle(toast).opacity || '0') > 0.02) return false;
  var sect = document.getElementById('mb-sect');
  if (sect && !sect.hidden) {
    var d = document.getElementById('mb-drive');
    var a = document.getElementById('mb-act');
    return !!d && d.className.indexOf('stale') < 0
      && !!a && a.textContent.trim().length > 1;
  }
  var bars = document.getElementById('ro-bars');
  return !bars || bars.className.indexOf('stale') < 0;
})()
"""

#: Read back what is on screen, so the check is on the page's own state rather
#: than on the pixels. Kept to one expression because CDP evaluates one.
_PROBE_JS = """
(function () {
  var c = window.flyClient;
  var q = function (id) { var e = document.getElementById(id); return e ? e.textContent.trim() : null; };
  var out = {
    head: q('head-mode'), footer: q('foot-text'),
    actions: q('st-act'), episodes: q('st-ep'), fps: q('st-fps'),
    mb_hidden: (document.getElementById('mb-sect') || {}).hidden,
    readout_hidden: (document.getElementById('ro-sect') || {}).hidden,
    verdict: q('mb-act'), dopamine: q('mb-da-text'), da_count: q('mb-da-count'),
    p_play: q('mb-p'), approach: q('mb-app'), avoid: q('mb-avo'),
    outcome: q('mb-outcome'), weights: q('mb-wstats'),
    buckets: Array.prototype.map.call(
      document.querySelectorAll('#mb-buckets .bk'),
      function (n) { return n.textContent.replace(/\\s+/g, ' ').trim(); }),
    reward: q('mb-rr-sub')
  };
  if (c && c.status) { out.policy = c.status.policy_mode; out.learning = c.status.learning; }
  return JSON.stringify(out);
})()
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=float, default=40.0,
                    help="real seconds to leave the page running before the shot")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--settle", type=float, default=12.0,
                    help="seconds to spend waiting for the decision panel to show "
                         "a finished hand before capturing; 0 captures blind")
    ap.add_argument("--devtools-port", type=int, default=9333)
    a = ap.parse_args(argv)

    if not CHROME.exists():
        print(f"no Chrome at {CHROME}", file=sys.stderr)
        return 2

    profile = Path(tempfile.mkdtemp(prefix="flyshot-"))
    proc = subprocess.Popen(
        [str(CHROME), f"--remote-debugging-port={a.devtools_port}",
         f"--user-data-dir={profile}", f"--window-size={a.width},{a.height}",
         *CHROME_FLAGS, "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = wait_for_devtools(a.devtools_port, time.time() + 30.0)
        result = asyncio.run(capture(ws_url, a.url, Path(a.out), a.wait,
                                     a.width, a.height, a.settle))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)

    out = Path(a.out)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    probe = result.get("probe")
    if probe:
        try:
            for k, v in json.loads(str(probe)).items():
                print(f"  {k}: {v}")
        except ValueError:
            print(f"  probe: {probe}")
    logs = result.get("logs") or []
    errors = [r for r in logs if not str(r["kind"]).endswith("warning")]
    warnings = [r for r in logs if str(r["kind"]).endswith("warning")]
    # Headless swiftshader narrates its own performance; that is the renderer
    # talking about itself, not the page misbehaving, so it is reported apart
    # from anything the page did.
    for label, rows in (("error", errors), ("warning", warnings)):
        if not rows:
            continue
        print(f"\n{len(rows)} console {label} entries:")
        for row in rows[:40]:
            print(f"  [{row['kind']}] {row['text'][:300]}")
    if not errors:
        print(f"\nno console errors ({len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
