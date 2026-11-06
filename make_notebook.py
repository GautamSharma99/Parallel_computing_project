#!/usr/bin/env python3
"""Builds Parallel_Image_Filters_Colab.ipynb: a self-contained Colab notebook that
writes all the source files, compiles them, benchmarks CPU vs GPU and draws the
charts. Re-run this after editing any source file:  python3 make_notebook.py"""
import json

SOURCES = ["common.h", "serial.c", "openmp.c", "cuda.cu", "Makefile", "bench.sh", "tools.py", "plot.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


cells = [
    md("""
# Parallel Image Filters: Serial vs OpenMP vs CUDA

Gaussian blur + Sobel edge detection on HD → 8K images, timed on one CPU core,
on all CPU cores (OpenMP) and on the GPU (CUDA).

**Before running:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
"""),
    md("## 1. Check the hardware"),
    code("!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv\n!echo \"CPU cores: $(nproc)\""),
    md("## 2. Write the source files\n(Same files as in the project folder.)"),
]
for name in SOURCES:
    with open(name) as f:
        cells.append(code(f"%%writefile {name}\n" + f.read()))

cells += [
    md("""
## 3. Make the test images
Set `USE_MY_PHOTO = True` to upload your own photo (a real photo looks better on
the slides). Otherwise a synthetic test picture is generated.
"""),
    code("""
import subprocess
USE_MY_PHOTO = False

args = []
if USE_MY_PHOTO:
    from google.colab import files
    args = ["--photo", list(files.upload())[0]]
subprocess.run(["python3", "tools.py", "prepare", *args], check=True)
"""),
    md("## 4. Compile all three versions"),
    code("!make all cuda"),
    md("## 5. Benchmark everything\nSerial, OpenMP (static + dynamic), CUDA naive and CUDA shared-memory, on every image size."),
    code("!bash bench.sh all"),
    md("## 6. Check correctness\nEvery parallel output must be pixel-identical to the serial output."),
    code("!python3 tools.py compare"),
    md("""
## 7. Charts
Tip: Colab only has 2 CPU cores, so the OpenMP scaling charts look better when
run on your laptop (`bash bench.sh cpu`). The CPU-vs-GPU comparison below is fair
because everything ran on this same machine.
"""),
    code("""
!python3 plot.py results/all.csv
from IPython.display import Image, Markdown, display
import glob
for p in sorted(glob.glob("charts/*.png")):
    display(Image(p, width=800))
display(Markdown(open("charts/summary.md").read()))
"""),
    md("## 8. Before / after"),
    code("""
!python3 tools.py topng > /dev/null
display(Image("png/before_after_fullhd.png", width=1000))
"""),
    md("## 9. Download results, charts and images for your slides"),
    code("""
!zip -qr results.zip results charts png/before_after_*.png
from google.colab import files
files.download("results.zip")
"""),
    md("""
## 10. Live demo (for the presentation)
Run this one cell in front of the audience (~10 seconds). It needs sections 1-5
to have run first: do *Run all* before the presentation starts.
"""),
    code("""
import subprocess
from PIL import Image as PILImage

def run(*cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1].split(",")

gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu_name}    Image: 8K, 7680 x 4320 = 33.2 million pixels\\n")
for mode in ("naive", "tiled"):
    r = run("./cuda_filter", "images/8k.pgm", f"out/demo_{mode}.pgm", mode, "16", "5")
    blur, sobel, copy, total = map(float, r[4:8])
    name = "naive (global memory)" if mode == "naive" else "shared memory (tiled)"
    print(f"CUDA {name:22s} blur {blur:5.2f} ms | sobel {sobel:5.2f} ms | "
          f"copies {copy:5.2f} ms | total {total:5.2f} ms  ({100 * copy / total:.0f}% copying)")

run("./serial", "images/8k.pgm", "out/demo_serial.pgm", "1")
same = all(open("out/demo_serial.pgm", "rb").read() == open(f"out/demo_{m}.pgm", "rb").read()
           for m in ("naive", "tiled"))
print("\\nGPU output identical to the serial CPU output:", "YES" if same else "NO")
display(PILImage.open("out/demo_tiled.pgm").resize((1600, 900)))
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}}
with open("Parallel_Image_Filters_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote Parallel_Image_Filters_Colab.ipynb")
