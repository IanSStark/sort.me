#!/usr/bin/env python
"""
Simple Flask web front-end for card-sorting camera/OCR tests.
URL pattern:
  GET  /            → index page with Capture button
  POST /capture     → returns JSON {title, position, image_base64}
"""

from flask import Flask, jsonify, request, render_template_string
import base64
import cv2
import datetime as dt

from capture import capture_frame          # existing module
from ocr import extract_title              # existing module
from sorter import determine_zone          # we only need letter mapping

# ROI rectangle used by extract_title (same as before)
ROI = (100, 150, 600, 100)

app = Flask(__name__)

# --- helper ---------------------------------------------------------------
def letter_position(letter: str) -> int | None:
    """Return A→1 … Z→26 or None for bad input."""
    if letter and letter.isalpha():
        return ord(letter.upper()) - ord("A") + 1
    return None


def frame_to_base64_png(frame):
    """Convert a NumPy BGR image to base64-encoded PNG."""
    # BGR (OpenCV) → RGB for nicer display
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --- routes ---------------------------------------------------------------
INDEX_HTML = """
<!doctype html>
<title>Card-Sorter Test</title>
<style>
 body { font-family:sans-serif; margin:2rem; }
 #img-preview { max-width: 480px; border:1px solid #aaa; }
</style>
<h1>Trading-Card OCR Test</h1>

<button id="capture">Capture Card</button>
<p id="status"></p>
<img id="img-preview">

<script>
document.getElementById('capture').onclick = async () => {
    document.getElementById('status').textContent = "Capturing…";
    const resp = await fetch("/capture", {method:"POST"});
    if (!resp.ok) { alert("Request failed"); return; }
    const data = await resp.json();
    document.getElementById('img-preview').src = "data:image/png;base64," + data.image_base64;
    document.getElementById('status').textContent =
        `Title: "${data.title}" — Alphabet position: ${data.position ?? "?"}`;
};
</script>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/capture", methods=["POST"])
def capture():
    frame = capture_frame()
    title = extract_title(frame, ROI)
    pos   = letter_position(title[0]) if title else None
    img_b64 = frame_to_base64_png(frame)
    # Respond
    return jsonify({
        "title": title or "",
        "position": pos,
        "image_base64": img_b64,
        "timestamp": dt.datetime.utcnow().isoformat()
    })


if __name__ == "__main__":
    # listens on all interfaces so you can browse from laptop
    app.run(host="0.0.0.0", port=5000, threaded=True)
