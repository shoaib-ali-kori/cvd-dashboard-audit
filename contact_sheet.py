"""
Builds contact sheets (thumbnail grids) of the captured screenshots so they can be reviewed in one image.
Usage: python contact_sheet.py --csv urls.csv --out data/raw --per-sheet 24
Writes: contact_sheet_1.png, contact_sheet_2.png, ...
"""
import argparse, csv, json
from pathlib import Path
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="urls.csv")
ap.add_argument("--out", default="data/raw")
ap.add_argument("--per-sheet", type=int, default=24)
a = ap.parse_args()

TW, TH, LAB, COLS = 320, 200, 28, 6
rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8-sig")))

def tile(r):
    d = Path(a.out) / r["id"]
    mp = d / "meta_default.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    img = Image.new("RGB", (TW, TH + LAB), (235, 235, 235))
    dr = ImageDraw.Draw(img)
    png = d / "default.png"
    if png.exists() and meta.get("status") == "ok":
        im = Image.open(png).convert("RGB")
        top = im.crop((0, 0, min(im.width, 1440), min(im.height, 900)))   # first screen only
        top.thumbnail((TW, TH))
        img.paste(top, (0, LAB))
    else:
        dr.text((10, LAB + 80), meta.get("status", "not captured"), fill=(150, 0, 0))
    label = f"{r['id']} {r['platform'][:14]} reg={meta.get('n_regions','-')}"
    dr.rectangle([0, 0, TW, LAB], fill=(30, 30, 30))
    dr.text((6, 8), label, fill=(255, 255, 255))
    return img

n = a.per_sheet
for s in range(0, len(rows), n):
    chunk = rows[s:s + n]
    nrows = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TW, nrows * (TH + LAB)), (255, 255, 255))
    for i, r in enumerate(chunk):
        sheet.paste(tile(r), ((i % COLS) * TW, (i // COLS) * (TH + LAB)))
    name = f"contact_sheet_{s // n + 1}.png"
    sheet.save(name)
    print("wrote", name)