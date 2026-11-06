#!/usr/bin/env python3
"""Image helpers for the project. Needs Pillow:  pip install pillow

  python3 tools.py prepare [--photo my.jpg]  make test images in images/ (HD, Full HD, 4K, 8K)
  python3 tools.py compare                   check every parallel output matches the serial one
  python3 tools.py topng                     convert images/ and out/ to PNG (png/) for slides
"""
import argparse
import glob
import os
import random
import sys

from PIL import Image, ImageChops, ImageDraw, ImageOps

SIZES = {"hd": (1280, 720), "fullhd": (1920, 1080), "4k": (3840, 2160), "8k": (7680, 4320)}


def synthetic(w, h, seed=7):
    """A test picture with lots of edges: shapes on a gradient, plus some noise."""
    rnd = random.Random(seed)
    img = Image.linear_gradient("L").resize((w, h))
    draw = ImageDraw.Draw(img)
    s = w / 1920
    for _ in range(80):
        x, y = rnd.randrange(w), rnd.randrange(h)
        r = int(rnd.uniform(20, 220) * s)
        fill = rnd.randrange(256)
        shape = rnd.choice(["ellipse", "rectangle", "line"])
        if shape == "line":
            end = (x + rnd.uniform(-500, 500) * s, y + rnd.uniform(-500, 500) * s)
            draw.line([(x, y), end], fill=fill, width=max(2, int(8 * s)))
        else:
            getattr(draw, shape)([x - r, y - r, x + r, y + r], fill=fill)
    noise = Image.effect_noise((w, h), 30)
    return Image.blend(img, noise, 0.12)


def prepare(args):
    os.makedirs("images", exist_ok=True)
    if args.photo:
        base = Image.open(args.photo).convert("L")
        print(f"Using {args.photo} ({base.width}x{base.height})")
    else:
        print("No --photo given, generating a synthetic test picture")
        base = synthetic(*SIZES["8k"])
    for name, size in SIZES.items():
        path = f"images/{name}.pgm"
        ImageOps.fit(base, size, Image.Resampling.LANCZOS).save(path)
        print(f"  {path}  {size[0]}x{size[1]}")


def compare(args):
    refs = sorted(glob.glob("out/*_serial.pgm"))
    if not refs:
        sys.exit("No out/*_serial.pgm yet - run: bash bench.sh cpu (or all)")
    all_ok = True
    for ref in refs:
        name = os.path.basename(ref)[: -len("_serial.pgm")]
        a = Image.open(ref)
        for other in sorted(glob.glob(f"out/{name}_*.pgm")):
            if other == ref:
                continue
            b = Image.open(other)
            if a.size != b.size:
                ok, msg = False, f"different size {b.size}"
            else:
                worst = ImageChops.difference(a, b).getextrema()[1]
                ok, msg = worst == 0, "identical" if worst == 0 else f"max pixel difference {worst}"
            all_ok &= ok
            print(f"{'OK  ' if ok else 'FAIL'} {other:34s} vs serial: {msg}")
    print("All outputs match the serial version." if all_ok else "Some outputs differ!")
    sys.exit(0 if all_ok else 1)


def topng(args):
    os.makedirs("png", exist_ok=True)
    for path in sorted(glob.glob("images/*.pgm") + glob.glob("out/*.pgm")):
        prefix = "input_" if path.startswith("images") else ""
        dest = f"png/{prefix}{os.path.basename(path)[:-4]}.png"
        Image.open(path).save(dest)
        print(dest)
    # Side-by-side "before | after" picture for the slides.
    for name in SIZES:
        src, edges = f"images/{name}.pgm", f"out/{name}_serial.pgm"
        if os.path.exists(src) and os.path.exists(edges):
            a, b = Image.open(src), Image.open(edges)
            pair = Image.new("L", (a.width * 2 + 20, a.height), 255)
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width + 20, 0))
            pair.save(f"png/before_after_{name}.png")
            print(f"png/before_after_{name}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare", help="create the test images")
    sp.add_argument("--photo", help="use this photo instead of a synthetic picture")
    sub.add_parser("compare", help="verify outputs against the serial version")
    sub.add_parser("topng", help="convert results to PNG")
    args = p.parse_args()
    {"prepare": prepare, "compare": compare, "topng": topng}[args.cmd](args)
