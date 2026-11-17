"""
Summarises what capture.py saved: one line per dashboard plus totals by platform.
Usage: python diagnose.py --csv urls.csv --out data/raw
"""
import argparse, csv, json
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="urls.csv")
ap.add_argument("--out", default="data/raw")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8-sig")))
tot = Counter()
blocked_hosts = Counter()
print(f"{'id':>4} {'platform':<9} {'status':<20} {'http':>4} {'reg':>3} {'ifr':>3} {'svg':>3} {'cvs':>3} {'KB':>5}  title")
for r in rows:
    d = Path(a.out) / r["id"]
    mp = d / "meta_default.json"
    if not mp.exists():
        mp = d / "meta.json"          # older v1 files
    if not mp.exists():
        print(f"{r['id']:>4} {r['platform']:<9} {'NOT CAPTURED':<20}")
        tot[(r["platform"], "not_captured")] += 1
        continue
    m = json.loads(mp.read_text())
    st = m.get("status", "?")
    html = ""
    hp = d / "page_default.html"
    if hp.exists():
        html = hp.read_text(encoding="utf-8", errors="ignore").lower()
    key = st if st != "ok" else ("ok" if m.get("n_regions", 0) > 0 else "ok_zero_regions")
    tot[(r["platform"], key)] += 1
    if st == "blocked_by_robots":
        blocked_hosts[(urlparse(m["url"]).netloc, m.get("robots", ""))] += 1
    print(f"{r['id']:>4} {r['platform']:<9} {st[:20]:<20} {str(m.get('http_status','')):>4} "
          f"{m.get('n_regions',''):>3} {html.count('<iframe'):>3} {html.count('<svg'):>3} {html.count('<canvas'):>3} "
          f"{len(html)//1024:>5}  {(m.get('title') or '')[:50]}")
print("\nTOTALS (platform, outcome) -> count")
for k, v in sorted(tot.items()):
    print(k, v)

print("\nBLOCKED HOSTS (host, reason) -> count")
for k, v in sorted(blocked_hosts.items(), key=lambda x: -x[1]):
    print(k, v)
print("\nOVERLAYS (id: clicked / hidden)")
for r in rows:
    mp = Path(a.out) / r["id"] / "meta_default.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        if m.get("status") == "ok":
            print(f"{r['id']:>4}: {m.get('overlays_clicked','-')} / {m.get('overlays_hidden','-')}")