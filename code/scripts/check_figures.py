"""Check a figure folder against the figure requirements of the manuscript.

  python code/scripts/check_figures.py [folder] [Fig1 Fig2 ...] [--json]      (default folder: figures)

Without figure names it checks the current set: Fig1 to Fig7 and S1_Fig to S4_Fig. For a folder that holds another
set, name the figures.

For each figure it reads the PDF (text as drawn: size, font, position) and the TIFF (pixels) and reports
  - text smaller than 8 pt or larger than 12 pt (this includes sub- and superscripts and tick exponents)
  - fonts other than Arial, Times or Symbol
  - pairs of text items whose boxes overlap
  - size: width 2.63 to 7.5 in (789 to 2250 px at 300 dpi), height at most 8.75 in (2625 px)
  - TIFF: RGB without alpha, 300 dpi, LZW, under 10 MB
Exit status 1 if any figure fails. Needs PyMuPDF and Pillow.
"""
import json
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
MIN_PT, MAX_PT, MIN_SCRIPT_PT = 8.0, 12.0, 6.0
MAX_W_PX, MIN_W_PX, MAX_H_PX = 2250, 789, 2625       # 7.5 x 8.75 in at 300 dpi
TOL = 0.06
OK_FONTS = ("arial", "times", "symbol")


def spans(pdf):
    page = fitz.open(pdf)[0]
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for sp in line["spans"]:
                if sp["text"].strip():
                    out.append({"text": sp["text"].strip(), "size": round(sp["size"], 2), "font": sp["font"],
                                "bbox": [round(v, 1) for v in sp["bbox"]], "dir": line.get("dir", (1, 0))})
    return out, page.rect.width / 72, page.rect.height / 72


def overlap(a, b):
    x = min(a[2], b[2]) - max(a[0], b[0])
    y = min(a[3], b[3]) - max(a[1], b[1])
    if x <= 0.6 or y <= 0.6:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return x * y / small if small > 0 else 0.0


def tight(sp):
    """Shrink a span's box to roughly its ink: the PDF box includes ascent and descent padding."""
    x0, y0, x1, y1 = sp["bbox"]
    horizontal = abs(sp["dir"][0]) > 0.7
    pad = 0.22 * sp["size"]
    return [x0, y0 + pad, x1, y1 - pad] if horizontal else [x0 + pad, y0, x1 - pad, y1]


def check(folder, name):
    res = {"figure": name, "problems": []}
    pdf, tif = folder / f"{name}.pdf", folder / f"{name}.tif"
    if not pdf.exists() or not tif.exists():
        res["problems"].append("missing " + ", ".join(p.name for p in (pdf, tif) if not p.exists()))
        return res
    sp, w_in, h_in = spans(pdf)
    sizes = sorted(s["size"] for s in sp)
    res.update(width_in=round(w_in, 2), height_in=round(h_in, 2), n_text=len(sp),
               min_pt=sizes[0] if sizes else None, max_pt=sizes[-1] if sizes else None)
    small = [s for s in sp if s["size"] < MIN_PT - TOL]
    scripts = []
    big = [s for s in sp if s["size"] > MAX_PT + TOL]
    badfont = sorted({s["font"] for s in sp if not any(k in s["font"].lower() for k in OK_FONTS)})
    if small:
        res["problems"].append(f"{len(small)} text items below {MIN_PT:g} pt: "
                               + "; ".join(f"{s['size']} pt '{s['text'][:40]}'" for s in small[:25]))
    if big:
        res["problems"].append(f"{len(big)} text items above {MAX_PT:g} pt: "
                               + "; ".join(f"{s['size']} pt '{s['text'][:40]}'" for s in big[:10]))
    if badfont:
        res["problems"].append("fonts other than Arial/Times/Symbol: " + ", ".join(badfont))
    res["short_items_6_to_8_pt"] = [f"{s['size']} pt '{s['text']}'" for s in scripts]
    pairs = []
    boxes = [tight(s) for s in sp]
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            if overlap(boxes[i], boxes[j]) > 0.08:
                pairs.append(f"'{sp[i]['text'][:30]}' x '{sp[j]['text'][:30]}'")
    if pairs:
        res["problems"].append(f"{len(pairs)} overlapping text pairs: " + "; ".join(pairs[:20]))
    im = Image.open(tif)
    dpi = im.info.get("dpi", (0, 0))
    res.update(tif_px=list(im.size), tif_mode=im.mode, tif_dpi=[round(float(d)) for d in dpi],
               tif_mb=round(tif.stat().st_size / 1e6, 2), tif_compression=im.info.get("compression"))
    max_h = MAX_H_PX
    if not (MIN_W_PX <= im.size[0] <= MAX_W_PX):
        res["problems"].append(f"TIFF width {im.size[0]} px outside {MIN_W_PX} to {MAX_W_PX} px")
    if im.size[1] > max_h:
        res["problems"].append(f"TIFF height {im.size[1]} px above {max_h} px")
    if im.mode != "RGB":
        res["problems"].append(f"TIFF mode {im.mode}, not RGB")
    if round(float(dpi[0])) != 300:
        res["problems"].append(f"TIFF resolution {dpi}, not 300 dpi")
    if im.info.get("compression") != "tiff_lzw":
        res["problems"].append(f"TIFF compression {im.info.get('compression')}, not LZW")
    if tif.stat().st_size > 10e6:
        res["problems"].append("TIFF larger than 10 MB")
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = ROOT / "figures"
    if args and ("/" in args[0] or "\\" in args[0] or (ROOT / args[0]).is_dir()):
        folder = Path(args.pop(0))
        folder = folder if folder.is_absolute() else ROOT / folder
    names = args or [f"Fig{i}" for i in range(1, 8)] + [f"S{i}_Fig" for i in range(1, 5)]
    results = [check(folder, n) for n in names]
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    else:
        for r in results:
            print(f"{r['figure']}: {r.get('width_in')} x {r.get('height_in')} in, text {r.get('min_pt')} to "
                  f"{r.get('max_pt')} pt ({r.get('n_text')} items), TIFF {r.get('tif_px')} {r.get('tif_mode')} "
                  f"{r.get('tif_mb')} MB -> " + ("OK" if not r["problems"] else "FAIL"))
            for p in r["problems"]:
                print("    -", p)
            if r.get("short_items_6_to_8_pt"):
                print("    note, short items between 6 and 8 pt (sub/superscripts?):", r["short_items_6_to_8_pt"][:12])
    sys.exit(1 if any(r["problems"] for r in results) else 0)
