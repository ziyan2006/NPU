"""Check RTL trace invariants and the teaching page in a real browser.

Needs Playwright and Chromium. Starts a temporary loopback-only HTTP server.
Use --screenshot to refresh the repository preview image after review.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import http.server
import io
import json
import shutil
import struct
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def check_trace():
    data = json.loads((HERE / "trace.js").read_text()[len("window.NPU_TRACE="):-2])
    rows = [dict(zip(["cycle"] + data["signals"], row)) for row in data["rows"]]
    def n(row, key):
        return None if row[key] is None else int(row[key], 16)
    def handshake(row, prefix):
        return n(row, prefix + "_v") == n(row, prefix + "_r") == 1
    # A freshly generated trace must correspond to the current source RTL.
    sources = (ROOT / "hardware/rtl/npu_rtl.f").read_text().splitlines()
    digest = hashlib.sha256(b"".join((ROOT / s).read_bytes() for s in sources if s and not s.startswith("#"))).hexdigest()
    assert data["rtlSha256"] == digest, "Recorded RTL differs from repository RTL"
    assert len(rows) == 1596
    assert all(b["cycle"] == a["cycle"] + 1 for a, b in zip(rows, rows[1:])), "Clock gap in trace"
    assert set(c["opcode"] for c in data["commands"]) == {0, 1, 3, 16, 17, 32, 48, 66}
    captures = [r for r in rows if n(r,"capture") == 1]
    assert [n(r,"fetch_pc") for r in captures] == list(range(0, 14*16, 16))
    assert n(rows[-1],"retired") == 14 and n(rows[-1],"irq") == 1
    mac_input = next(i for i, r in enumerate(rows) if handshake(r,"mac"))
    for delta, key in enumerate(["mac_product","mac_l1","mac_l2","mac_dot","mac_out_v"], 1):
        assert n(rows[mac_input+delta],key) == 1, f"MAC latency mismatch: {key}"
    assert int(rows[mac_input+5]["mac_data"],16) == 6
    o_write = next(i for i,r in enumerate(rows) if handshake(r,"o"))
    conv_done = next(i for i,r in enumerate(rows) if n(r,"conv_event") == 16)
    assert int(rows[o_write]["o_data"],16) == 6 and conv_done > o_write
    assert n(rows[conv_done],"fifo_count") == 0
    vec_write = next(r for r in rows if handshake(r,"v_write"))
    assert int(vec_write["v_write_data"],16) == 9
    # Decode the accepted AXI output beats: one store pixel, then four copies.
    writes = [int(r["w_data"],16) for r in rows if handshake(r,"w")]
    assert b"".join(struct.pack("<Q",v) for v in writes) == (struct.pack("<h",9)+bytes(14))*5
    assert any(n(r,"dma_v")==1 and n(r,"dma_r")==0 for r in rows), "No dispatch backpressure captured"
    print("RTL trace: continuous clocks, all 8 opcodes, MAC latency, FIFO drain and DDR output PASS")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def check_browser(screenshot=False):
    from playwright.sync_api import sync_playwright
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/teaching/"
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    try:
        with sync_playwright() as p, tempfile.TemporaryDirectory() as temp:
            browser = p.chromium.launch(executable_path=chromium, headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width":1440,"height":1100},accept_downloads=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base)
            page.wait_for_function("window.NPULab")
            assert page.locator("#instruction option").count() == 8
            assert page.locator("#program button").count() == 14
            assert page.locator(".module").count() == 25
            for name in ["NOP","WAIT","END","DMA_LOAD","DMA_STORE","CONV2D","VEC_ADD","UPSAMPLE2X"]:
                page.select_option("#instruction",name)
                for option in page.locator("#case option").all():
                    index = option.get_attribute("value")
                    page.select_option("#case",index)
                    assert page.evaluate("NPULab.position === NPULab.selection.start")
                    assert page.locator("#edge-fetch_cp").get_attribute("class") == "edge active"
                    page.locator("#seek").evaluate("e=>{e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}))}")
                    assert page.evaluate("NPULab.position === NPULab.selection.end")
                    assert page.locator("#next").is_disabled()
                    page.click("#reset")
            page.select_option("#instruction","CONV2D")
            page.click("#next")
            assert page.evaluate("NPULab.position === NPULab.selection.start+1")
            assert "active" in page.locator("#edge-cp_ef").get_attribute("class")
            page.click("#key-next")
            assert page.evaluate("NPULab.position > NPULab.selection.start+1")
            page.click("#node-mac")
            assert page.locator("#module-title").inner_text() == "8 × 8 MAC"
            page.click("#help-button")
            assert page.locator("#guide").is_visible()
            page.click("#help-button")
            page.select_option("#diagram-zoom","125")
            assert page.locator("#diagram").evaluate("e=>e.getBoundingClientRect().width") == 1750
            page.select_option("#diagram-zoom","fit")
            page.select_option("#speed","30")
            page.click("#play")
            old = page.evaluate("NPULab.position")
            page.wait_for_function("NPULab.position > " + str(old+2))
            page.click("#play")
            assert page.locator("#play").get_attribute("aria-pressed") == "false"
            # Wave cells and scrubber must point to the same real cycle.
            page.locator("#wave .wave-hit").first.click()
            assert page.evaluate("Number(document.querySelector('#seek').value) === NPULab.position - NPULab.selection.start")
            with page.expect_download() as event:
                page.click("#export-trace")
            csv_path = Path(temp)/"trace.csv"
            event.value.save_as(csv_path)
            table=list(csv.reader(io.StringIO(csv_path.read_text())))
            assert len(table)==page.evaluate("NPULab.selection.end-NPULab.selection.start+2")
            assert "task_cycle" in table[0] and table[1][0]=="0"
            with page.expect_download() as event:
                page.click("#download-svg")
            svg_path = Path(temp)/"diagram.svg"
            event.value.save_as(svg_path)
            ET.parse(svg_path)
            assert "rgb(89, 221, 220)" in svg_path.read_text(), "SVG lost its active colors"
            # Restore a useful MAC-result frame for the repository preview.
            page.click("#reset")
            offset = page.evaluate("NPULab.frames.findIndex((f,i)=>i>=NPULab.selection.start && f[NPULab.signals.mac_out_v]==='1')-NPULab.selection.start")
            page.locator("#seek").evaluate("(e,value)=>{e.value=value;e.dispatchEvent(new Event('input',{bubbles:true}))}",offset)
            if screenshot:
                page.select_option("#speed","8")
                page.wait_for_function("!document.querySelector('#toast').classList.contains('visible')")
                page.screenshot(path=str(HERE/"preview.png"),full_page=True)
            assert "6" in page.locator("#edge-detail").inner_text()
            # Verify a held VALID is represented by waiting, never transfer.
            page.select_option("#instruction","DMA_LOAD")
            page.select_option("#case","2")
            offset=page.evaluate("NPULab.frames.findIndex((f,i)=>i>=NPULab.selection.start&&f[NPULab.signals.dma_v]==='1'&&f[NPULab.signals.dma_r]==='0')-NPULab.selection.start")
            page.locator("#seek").evaluate("(e,value)=>{e.value=value;e.dispatchEvent(new Event('input',{bubbles:true}))}",offset)
            assert page.locator("#edge-cp_df").get_attribute("class") == "edge wait"
            # The standalone page loads no scripts, styles or data from HTTP.
            standalone = browser.new_page(viewport={"width":1440,"height":1100})
            standalone.on("pageerror",lambda error: errors.append(str(error)))
            requests=[]
            standalone.on("request",lambda request:requests.append(request.url))
            standalone.goto(base+"npu-lab.html")
            standalone.wait_for_function("window.NPULab")
            assert requests==[base+"npu-lab.html"]
            standalone.select_option("#instruction","END")
            standalone.locator("#seek").evaluate("e=>{e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}))}")
            assert "irq_o = 1" in standalone.locator("#node-irq").text_content()
            standalone.set_viewport_size({"width":390,"height":844})
            assert standalone.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile page overflows"
            assert not errors, errors
            browser.close()
            print("Browser: all 14 cases, stepping, playback, seek, wave, module, exports, offline bundle and mobile layout PASS")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot",action="store_true")
    args=parser.parse_args()
    check_trace()
    check_browser(args.screenshot)
