"""Record the demo GIF: uv run --with pillow==13.0.0 python docs/record_demo_gif.py [font.ttf]"""

import hashlib
import sys
from io import StringIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mloda.interfaces.cli.main import main

COMMAND = "$ uvx mloda@latest demo\n"
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG = "#1d1f27"
FG = "#d4d4d8"
DOT = "#5c6370"
PAD = 18
SIZE = 15
BAR = 30


def record() -> None:
    buf = StringIO()
    code = main(["demo"], stdout=buf)
    if code != 0:
        sys.exit(f"mloda demo exited with code {code}")
    text = COMMAND + buf.getvalue()

    font_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FONT)
    if not font_path.is_file():
        sys.exit(f"font not found: {font_path}; pass a monospace .ttf as the first argument")
    font = ImageFont.truetype(str(font_path), SIZE)
    assets = Path(__file__).parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    lines = text.rstrip("\n").split("\n")
    line_h = round(SIZE * 1.35)
    char_w = font.getlength("M")
    width = int(max(len(line) for line in lines) * char_w) + 2 * PAD
    height = BAR + PAD + len(lines) * line_h + PAD

    def frame(shown: str) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(img)
        for i in range(3):
            x = PAD + i * 18
            draw.ellipse((x, 10, x + 10, 20), fill=DOT)
        for n, line in enumerate(shown.split("\n")):
            draw.text((PAD, BAR + n * line_h), line, font=font, fill=FG)
        return img

    typed = COMMAND.rstrip("\n")
    final = frame(text.rstrip("\n")).quantize(colors=32)
    frames = [frame("$ ").quantize(palette=final)]
    durations = [800]
    for i in range(3, len(typed) + 1):
        frames.append(frame(typed[:i]).quantize(palette=final))
        durations.append(70)
    durations[-1] = 500
    frames.append(final)
    durations.append(8000)

    frames[0].save(
        assets / "mloda-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        comment=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


if __name__ == "__main__":
    record()
