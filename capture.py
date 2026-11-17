"""
Step 1 - Dashboard capture (Phase 3), v2.
Reads urls.csv (id,url,platform,domain) and saves, per dashboard, in data/raw/<id>/:
  <theme>.png            full-page screenshot, 1440x900 viewport
  page_<theme>.html      rendered HTML of the top document
  regions_<theme>.json   chart-like regions (svg/canvas/img/iframe) from the page AND its iframes
  meta_<theme>.json      url, timestamp, status, reason (also written for blocked/failed pages)
Respects robots.txt, rate-limits requests, resumes on re-run.

Usage:
  python capture.py --csv urls.csv --out data/raw [--theme default|dark] [--force] [--ids 6,11,12]
"""
import argparse, csv, json, re, time, sys, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

UA = "Mozilla/5.0 (compatible; DAVResearchBot/1.0; academic research, FAST-NUCES)"

# arg = true for the top document (add page scroll), false inside iframes
JS_REGIONS = """
(isTop) => {
  const out = [];
  const sx = isTop ? window.scrollX : 0, sy = isTop ? window.scrollY : 0;
  document.querySelectorAll('svg, canvas, img, iframe').forEach((el, i) => {
    const r = el.getBoundingClientRect();
    if (r.width < 150 || r.height < 100) return;
    const tag = el.tagName.toLowerCase();
    const rec = {idx: i, tag: tag, x: r.left + sx, y: r.top + sy, w: r.width, h: r.height,
                 aria: el.getAttribute('aria-label') || el.getAttribute('alt') || el.getAttribute('title') || ''};
    if (tag === 'svg') {
      const fills = {}, strokes = {}, shapes = {};
      el.querySelectorAll('rect,path,circle,polygon,ellipse,line,use').forEach(m => {
        const cs = getComputedStyle(m);
        const t = m.tagName.toLowerCase();
        shapes[t] = (shapes[t] || 0) + 1;
        if (cs.fill && cs.fill !== 'none') fills[cs.fill] = (fills[cs.fill] || 0) + 1;
        if (cs.stroke && cs.stroke !== 'none') strokes[cs.stroke] = (strokes[cs.stroke] || 0) + 1;
      });
      rec.fills = fills; rec.strokes = strokes; rec.shapes = shapes;
      rec.n_text = el.querySelectorAll('text').length;
      rec.n_pattern = el.querySelectorAll('pattern').length;
      rec.has_gradient = el.querySelectorAll('linearGradient,radialGradient').length > 0;
    }
    out.push(rec);
  });
  return out;
}
"""

_robots_cache = {}

def robots_check(url):
    """Returns (allowed, reason). Uses an explicit User-Agent so CDNs do not 403 the default Python agent."""
    p = urlparse(url)
    key = f"{p.scheme}://{p.netloc}"
    if key not in _robots_cache:
        robots_url = key + "/robots.txt"
        rp = robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                rp.parse(resp.read().decode("utf-8", "ignore").splitlines())
            _robots_cache[key] = (rp, "robots_read")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                _robots_cache[key] = (None, f"robots_http_{e.code}")   # conservative: treat as blocked
            else:
                _robots_cache[key] = (None, f"no_robots_http_{e.code}")  # 404 etc: no restrictions published
        except Exception as e:
            _robots_cache[key] = (None, f"robots_unreachable_{type(e).__name__}")  # treat as allowed
    rp, reason = _robots_cache[key]
    if reason.startswith("robots_http_"):
        return False, reason
    if rp is None:
        return True, reason
    return rp.can_fetch("DAVResearchBot", url), "robots_disallow" if not rp.can_fetch("DAVResearchBot", url) else reason


DISMISS_RE = re.compile(
    r"^(accept only essential cookies|only essential( cookies)?|reject all|accept all( cookies)?|accept( cookies)?|"
    r"i agree|agree|got it|ok|okay|close|dismiss|no thanks|maybe later)$", re.I)

# elements that look like pop-ups, cookie banners or dim backdrops (Playwright CSS also reaches open shadow DOM)
OVERLAY_SEL = ('[role="dialog"], [role="alertdialog"], [aria-modal="true"], [class*="modal" i], [class*="overlay" i], '
               '[class*="backdrop" i], [class*="cookie" i], [class*="consent" i], [class*="popup" i], '
               '[id*="cookie" i], [id*="consent" i], [id*="modal" i]')

JS_PROBE = """els => els.map(e => {
  const cs = getComputedStyle(e), r = e.getBoundingClientRect();
  return {tag: e.tagName.toLowerCase(), id: (e.id || '').slice(0, 40),
          cls: ((e.className && e.className.toString()) || '').slice(0, 80),
          role: e.getAttribute('role') || '', pos: cs.position, z: parseInt(cs.zIndex) || 0,
          w: Math.round(r.width), h: Math.round(r.height),
          vis: cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0,
          text: (e.innerText || '').trim().slice(0, 100),
          buttons: Array.from(e.querySelectorAll('button,a,[role=button]')).slice(0, 6).map(b => (b.innerText || b.getAttribute('aria-label') || '').trim().slice(0, 40))};
}).filter(o => o.vis && o.pos !== 'static')"""

JS_HIDE = """els => { let n = 0;
  els.forEach(e => {
    const cs = getComputedStyle(e), r = e.getBoundingClientRect();
    if (cs.display === 'none' || r.width === 0) return;
    const z = parseInt(cs.zIndex) || 0, role = e.getAttribute('role');
    const modalish = role === 'dialog' || role === 'alertdialog' || e.getAttribute('aria-modal') === 'true';
    if ((cs.position === 'fixed' || cs.position === 'absolute') && (z >= 10 || modalish)) {
      e.style.setProperty('display', 'none', 'important'); n++; }
  });
  document.documentElement.style.setProperty('overflow', 'auto', 'important');
  return n; }"""

def dismiss_overlays(page):
    """Best effort: close pop-ups / cookie banners so they do not cover the dashboard.
    Order: Escape -> click an accept/close control by its text -> hide fixed dialog/backdrop elements.
    Two passes, because some pop-ups appear a few seconds after load. Returns a dict recorded in meta."""
    info = {"overlays_clicked": 0, "overlays_hidden": 0, "overlay_probe": []}
    for p in range(2):
        try:
            time.sleep(2.0 if p == 0 else 3.0)
            if p == 0:
                try:
                    info["overlay_probe"] = page.locator(OVERLAY_SEL).evaluate_all(JS_PROBE)[:8]
                except Exception:
                    pass
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            for _ in range(3):
                clicked = False
                for loc in (page.get_by_role("button", name=DISMISS_RE),
                            page.get_by_text(DISMISS_RE),
                            page.locator('[aria-label="Close" i], button[title="Close" i]')):
                    try:
                        for i in range(min(loc.count(), 4)):
                            el = loc.nth(i)
                            if el.is_visible():
                                el.click(timeout=1500)
                                info["overlays_clicked"] += 1
                                clicked = True
                                break
                    except Exception:
                        pass
                    if clicked:
                        break
                if not clicked:
                    break
                time.sleep(1.0)
            try:
                info["overlays_hidden"] += page.locator(OVERLAY_SEL).evaluate_all(JS_HIDE)
            except Exception:
                pass
        except Exception:
            pass
    return info

def collect_regions(page):
    regions = []
    top = page.evaluate(JS_REGIONS, True)
    for r in top:
        r["frame"] = "top"
    regions += top
    sx, sy = page.evaluate("[window.scrollX, window.scrollY]")
    for fr in page.frames:
        if fr == page.main_frame:
            continue
        try:
            fe = fr.frame_element()
            bb = fe.bounding_box()
            if not bb or bb["width"] < 150 or bb["height"] < 100:
                continue
            inner = fr.evaluate(JS_REGIONS, False)
        except Exception:
            continue
        for r in inner:
            r["x"] += bb["x"] + sx
            r["y"] += bb["y"] + sy
            r["frame"] = fr.url[:200]
        regions += inner
    return regions

def capture_one(browser, row, out_root, theme, wait_s, force):
    d = Path(out_root) / row["id"]
    shot = d / f"{theme}.png"
    if shot.exists() and not force:
        return "skipped (already captured)"
    d.mkdir(parents=True, exist_ok=True)
    if force:   # remove files from earlier runs so a blocked page cannot keep an old screenshot
        for old in [shot, d / f"page_{theme}.html", d / f"regions_{theme}.json", d / f"meta_{theme}.json", d / "meta.json"]:
            old.unlink(missing_ok=True)
    meta = {"id": row["id"], "url": row["url"], "platform": row.get("platform", ""),
            "domain": row.get("domain", ""), "theme": theme,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    mpath = d / f"meta_{theme}.json"
    allowed, reason = robots_check(row["url"])
    meta["robots"] = reason
    if not allowed:
        meta["status"] = "blocked_by_robots"
        mpath.write_text(json.dumps(meta, indent=2))
        return f"blocked by robots.txt ({reason})"
    ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                              color_scheme="dark" if theme == "dark" else "light")
    page = ctx.new_page()
    try:
        resp = page.goto(row["url"], wait_until="networkidle", timeout=45000)
        meta["http_status"] = resp.status if resp else None
    except PWTimeout:
        meta["note"] = "networkidle timeout; captured whatever had rendered"
    except Exception as e:
        meta["status"] = f"error: {type(e).__name__}"
        mpath.write_text(json.dumps(meta, indent=2))
        ctx.close()
        return meta["status"]
    time.sleep(wait_s)
    meta.update(dismiss_overlays(page))
    time.sleep(1.5)
    page.screenshot(path=str(shot), full_page=True)
    (d / f"page_{theme}.html").write_text(page.content(), encoding="utf-8")
    regions = collect_regions(page)
    (d / f"regions_{theme}.json").write_text(json.dumps(regions, indent=2))
    meta["title"] = page.title()
    meta["n_frames"] = len(page.frames) - 1
    meta["n_regions"] = len(regions)
    meta["status"] = "ok"
    mpath.write_text(json.dumps(meta, indent=2))
    ctx.close()
    return f"ok ({len(regions)} regions, {meta['n_frames']} iframes)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="urls.csv")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--theme", default="default", choices=["default", "dark"])
    ap.add_argument("--wait", type=float, default=4.0)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--force", action="store_true", help="re-capture even if a screenshot exists")
    ap.add_argument("--ids", default="", help="comma-separated ids to (re)capture, e.g. 6,11,12")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8-sig")))
    if a.ids:
        keep = {s.strip() for s in a.ids.split(",")}
        rows = [r for r in rows if r["id"] in keep]
    if not rows:
        sys.exit("No rows to capture. Check urls.csv (columns: id,url,platform,domain) and --ids.")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for n, row in enumerate(rows, 1):
            res = capture_one(browser, row, a.out, a.theme, a.wait, a.force)
            print(f"[{n}/{len(rows)}] {row['id']}: {res}", flush=True)
            time.sleep(a.delay)
        browser.close()

if __name__ == "__main__":
    main()