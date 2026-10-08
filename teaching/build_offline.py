"""Bundle the teaching page into one HTML file that can be opened offline."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    css = (HERE / "style.css").read_text(encoding="utf-8")
    trace = (HERE / "trace.js").read_text(encoding="utf-8")
    lesson = (HERE / "lesson.js").read_text(encoding="utf-8")
    app = (HERE / "app.js").read_text(encoding="utf-8")
    # Inline scripts go after the document, since inline `defer` is ignored.
    for src in ("trace.js", "lesson.js", "app.js"):
        html = html.replace(f'  <script src="{src}" defer></script>\n', "")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace("</body>", f"<script>\n{trace}\n</script>\n<script>\n{lesson}\n</script>\n<script>\n{app}\n</script>\n</body>")
    html = html.replace('href="README.md"', 'href="https://github.com/ziyan2006/NPU/blob/work/teaching/README.md"')
    html = html.replace('href="../框图/03_npu_internal_io.svg"', 'href="https://github.com/ziyan2006/NPU/blob/work/框图/03_npu_internal_io.svg"')
    output = HERE / "npu-lab.html"
    output.write_text(html, encoding="utf-8", newline="\n")
    print(f"Built {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
