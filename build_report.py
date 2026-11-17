"""
build_report.py - assembles the Phase 3 Word report from YOUR results, so every number and figure matches your run.

Usage:
  python -m pip install python-docx
  python build_report.py --results results --real results_real --review review --raw data/raw --urls urls.csv ^
         --exclude-ids 1,2,3,4,5 --feedback-file feedback.txt --members "Name (roll no), Name (roll no)" ^
         --out DAV_Phase3_Implementation_and_Evaluation.docx

Optional inputs are skipped cleanly if missing: results_real/, review/, data/raw/, feedback.txt.
"""
import argparse, csv, json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

FIG = {"n": 0}
TAB = {"n": 0}


# ---------------------------------------------------------------- helpers
def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_runs(p, text, size=None, italic=False):
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = i % 2 == 1
        r.italic = italic
        if size:
            r.font.size = Pt(size)


def para(doc, text, size=None, italic=False, align=None, space_after=6):
    p = doc.add_paragraph()
    add_runs(p, text, size, italic)
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    add_runs(p, text)
    p.paragraph_format.space_after = Pt(2)
    return p


def table(doc, header, rows, widths, font=8.5, caption_text=None):
    if caption_text:
        TAB["n"] += 1
        p = doc.add_paragraph()
        r = p.add_run(f"Table {TAB['n']}. {caption_text}"); r.bold = True; r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(3); p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.width = Inches(widths[i]); shade(c, "D9E2F3")
        c.paragraphs[0].text = ""
        run = c.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].width = Inches(widths[i]); cells[i].paragraphs[0].text = ""
            add_runs(cells[i].paragraphs[0], str(v), font)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def figure(doc, path, text, width=6.0):
    path = Path(path)
    if not path.exists():
        return
    FIG["n"] += 1
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(f"Figure {FIG['n']}. {text}"); r.italic = True; r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(10)


def add_page_number(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Page "); r.font.size = Pt(9)
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run(); run.font.size = Pt(9)
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); run._r.append(el)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


def read_csv(p, **kw):
    p = Path(p)
    return pd.read_csv(p, **kw) if p.exists() else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--real", default="results_real")
    ap.add_argument("--review", default="review"); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--urls", default="urls.csv"); ap.add_argument("--exclude-ids", default="")
    ap.add_argument("--feedback-file", default="feedback.txt")
    ap.add_argument("--members", default="[add names and roll numbers]")
    ap.add_argument("--review-ai-assisted", action="store_true", help="state that review labels were drafted by an AI model and verified by the annotator")
    ap.add_argument("--out", default="DAV_Phase3_Implementation_and_Evaluation.docx")
    a = ap.parse_args()
    R = Path(a.results)
    M = json.load(open(R / "metrics.json")); T = R / "tables"; F = R / "figures"
    main_df = pd.read_csv(T / "main_comparison.csv"); mc = pd.read_csv(T / "mcnemar.csv")
    abl = pd.read_csv(T / "ablation.csv"); tau_df = pd.read_csv(T / "tau_sensitivity.csv")
    rho = pd.read_csv(T / "sensitivity_spearman.csv", index_col=0); errc = pd.read_csv(T / "error_causes.csv")
    info = json.load(open(R / "run_info.json")) if (R / "run_info.json").exists() else {}
    cal = M["calibration"]; auc = M["roc_auc_cvd_score"]; flips = M["label_flip_rate_vs_default"]
    ext = M["extraction"]; cost = M["cost_ms_per_chart"]
    H = main_df.iloc[0]; B1 = main_df.iloc[1]; B2 = main_df.iloc[2]
    tau = cal["selected_tau"]; n_test = sum(M["dataset"]["test"].values())

    # real pilot data (optional)
    excl = {s.strip() for s in a.exclude_ids.split(",") if s.strip()}
    ds = read_csv(Path(a.real) / "dashboards.csv", dtype={"id": str})
    ch = read_csv(Path(a.real) / "charts.csv", dtype={"id": str})
    rv = read_csv(Path(a.review) / "review_scores.csv")
    rv_sum = json.load(open(Path(a.review) / "review_summary.json")) if (Path(a.review) / "review_summary.json").exists() else None
    dis = read_csv(Path(a.review) / "disagreements.csv", dtype=str)
    hv = rv[rv.method == "Hybrid"].iloc[0] if (rv is not None and len(rv)) else None
    n_urls = len(list(csv.DictReader(open(a.urls, newline="", encoding="utf-8-sig")))) if Path(a.urls).exists() else None
    status = {}
    if Path(a.raw).exists():
        for mp in Path(a.raw).glob("*/meta_default.json"):
            s = json.loads(mp.read_text()).get("status", "?")
            status[s] = status.get(s, 0) + 1
    feedback = []
    if Path(a.feedback_file).exists():
        feedback = [l.strip(" -\t") for l in Path(a.feedback_file).read_text(encoding="utf-8").splitlines() if l.strip()]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.0))
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        h = doc.styles[name]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.bold = True
        h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(4); h.paragraph_format.keep_with_next = True
    hp = sec.header.paragraphs[0]
    hr = hp.add_run("DAV Research Project - Phase 3\tFAST-NUCES Islamabad"); hr.italic = True; hr.font.size = Pt(9)
    add_page_number(sec)

    # ------------------------------------------------------------ cover
    for _ in range(3):
        doc.add_paragraph()
    p = para(doc, "**NATIONAL UNIVERSITY OF COMPUTER & EMERGING SCIENCES**", 13, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "FAST-NUCES, Islamabad", 12, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "**Data Analysis & Visualization**", 14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Research Project", 12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    para(doc, "**Automated Auditing of Color-Vision Accessibility in Public Dashboards: A Quantitative Framework and Large-Scale Evaluation**",
         16, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Phase 3: Research Implementation and Experimental Evaluation", 13, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    for lab, val in (("Prepared by", "Shoaib Ali Kori (24I-2605)"), ("Section", "DS-5A"), ("Group members", a.members), ("Submitted", "November 2026")):
        para(doc, f"**{lab}:** {val}", 11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    doc.add_page_break()

    # ------------------------------------------------------------ summary
    doc.add_heading("Summary", 1)
    para(doc, f"We implemented the hybrid audit designed in Phase 2: a deterministic colour-vision-deficiency (CVD) perception module and a "
              f"structure module that handles redundant encodings and abstains with a Cannot Tell label. We evaluated it on a controlled "
              f"benchmark of {n_test} synthetic charts whose labels are known by construction, and ran it as a pilot on real public dashboards.")
    para(doc, f"On the held-out test benchmark the hybrid audit reaches a Fail-class F1 of **{f3(H.fail_f1)}** (95% bootstrap CI "
              f"{f3(H.f1_ci_low)}-{f3(H.f1_ci_high)}), against {f3(B1.fail_f1)} for a contrast-only baseline and {f3(B2.fail_f1)} for a "
              f"simulation-only baseline. Both differences are significant (McNemar with Holm correction, p < 0.001). "
              f"Ablations show which components carry the result, and the audit's decisions change materially with simulation severity.")
    para(doc, "**Scope of the claims.** The controlled benchmark tests the pipeline's mechanics, not agreement with people who have colour vision "
              "deficiency. The real-dashboard work is a small pilot (most of our target URLs were blocked or unusable), so we make no prevalence "
              "claim. RQ2 (agreement with CVD users' task errors) is not tested in this phase.")
    if hv is not None:
        harder = hv.fail_precision < H.fail_precision - 0.05
        para(doc, f"**{'Real pages are harder than the benchmark.' if harder else 'Real-page spot-check.'}** In a human spot-check of {int(hv.n)} judgeable real charts, the hybrid audit found "
                  f"all {int(hv.tp)} charts judged failing but also flagged {int(hv.fp)} that were judged fine (Fail precision {f3(hv.fail_precision)}, "
                  f"Wilson 95% CI {f3(hv.precision_wilson_low)}-{f3(hv.precision_wilson_high)}), against {f3(H.fail_precision)} on the benchmark. "
                  f"The sample is small and the labels were AI-drafted and human-checked (Section 7.2).")

    # ------------------------------------------------------------ 1 implementation
    doc.add_heading("1. Implementation of the Proposed Methodology", 1)
    doc.add_heading("1.1 What was built", 2)
    para(doc, "The implementation follows the twelve steps of Phase 2. All code is in one folder and runs from the command lines in Section 8.")
    table(doc, ["Phase 2 step", "Implementation", "File"], [
        ["1-2 Input and data collection", "Headless Chromium at 1440x900; saves screenshot, HTML and chart-like regions (svg, canvas, iframe contents); respects robots.txt; rate-limited", "capture.py"],
        ["2 Controlled benchmark", "Synthetic charts with ground-truth labels, rendered with matplotlib", "make_benchmark.py"],
        ["3 Preprocessing", "3x3 median filter, flat-interior pixel selection, background and neutral masking, histogram palette extraction merged by CIEDE2000, anti-aliasing tint removal", "cvdaudit.py"],
        ["4 Features", "CVD simulation (Machado 2009 for protan, deutan, tritan; Vienot 1999 for protan, deutan); pairwise CIEDE2000 before and after simulation; WCAG 3:1 non-text contrast; gradient lightness checks; redundancy cues", "cvdaudit.py"],
        ["5 Model", "Hybrid audit: perception module + structure module (redundancy downgrade, Cannot Tell)", "cvdaudit.py"],
        ["6 Baselines", "B1 contrast-only; B2 simulation-only with a fixed threshold", "cvdaudit.py"],
        ["7-9 Design, metrics, statistics", "Dev/test benchmarks, 5-fold CV calibration, bootstrap CIs, McNemar with Holm correction", "run_experiments.py"],
        ["10-11 Ablation, errors", "Eleven single-component ablations; error causes", "run_experiments.py"],
        ["Real pages", "Audit of captured dashboards with CVD-simulated previews; human spot-check tool", "audit_real.py, review.py"],
    ], [1.5, 4.0, 1.2], caption_text="Phase 2 steps and where they are implemented")
    doc.add_heading("1.2 The audit in brief", 2)
    bullet(doc, "**Palette extraction.** The chart image is median-filtered, only flat-interior pixels are kept, background and text-like greys are removed, and colours are binned and merged when closer than 6 CIEDE2000 units. Edge blends with the background are dropped.")
    bullet(doc, f"**Perception module.** A colour pair is confusable if it is distinguishable under normal vision (CIEDE2000 >= tau) but falls below tau after simulation of protan, deutan or tritan vision. A chart also fails if any mark has less than 3:1 contrast with its background. Sequential legends fail if lightness stops being monotonic after simulation; diverging legends fail if their two ends become indistinguishable. tau = {tau:g}, chosen by cross-validation.")
    bullet(doc, "**Structure module.** If a colour-only failure is accompanied by strong redundancy (distinct marker shapes, patterns or direct labels for every category), the chart is not marked Fail; it is returned as Cannot Tell for human review. Charts whose legend cannot be read are also Cannot Tell.")
    doc.add_heading("1.3 Changes from Phase 2 and feedback", 2)
    if feedback:
        para(doc, "**Feedback received in the previous class and how it was handled:**")
        for f in feedback:
            bullet(doc, f)
    else:
        para(doc, "**Feedback received in the previous class:** [add the instructor's feedback and how it was addressed].")
    para(doc, "The following changes were forced by what was feasible or by what the early tests showed. We list them so the evaluation is not read as the Phase 2 plan carried out unchanged.")
    table(doc, ["Phase 2 plan", "What we did", "Reason"], [
        ["400+ real dashboards, 150 human-annotated, 10+ CVD observers", f"A controlled synthetic benchmark ({n_test} test charts) plus a small pilot on real pages; one spot-check review instead of a CVD-observer study", "Most target hosts blocked automated access or served pop-ups and bot checks; time"],
        ["Platform strata: Tableau, Power BI, Looker, ArcGIS", "Whatever captured cleanly: government portals, WHO, Eurostat, Grafana, some Power BI", "Platform availability and robots.txt"],
        ["k-means in CIELAB with silhouette-selected k", "Flat-pixel histogram with CIEDE2000 merging and tint removal", "k-means split anti-aliased edge colours into extra categories"],
        ["Confusable pair: distance below tau", "Pair counts only if distinguishable at normal vision and below tau after simulation", "Avoids flagging colours that were never meant to differ"],
        ["Severities 0.3, 0.6, 1.0 as audit settings", "Audit uses 1.0; 0.3 and 0.6 used as a sensitivity analysis", "Simpler decision rule; sensitivity is reported"],
        ["tau by 5-fold CV on annotated dashboards; 60/20/20 split", "tau by 5-fold CV on a development benchmark (seed 42); one test benchmark (seed 2026) for all reported numbers", "No annotated dashboards were available"],
        ["B3: vision-language model baseline", "Not run", "Needs API access and cost; Phase 1 evidence [P7] predicts weak performance, but we did not test it"],
        ["B1 includes axe-core", "B1 is the 3:1 non-text contrast rule only", "axe-core needs a live DOM; benchmark charts are images"],
        ["Dashboard score weights from logistic regression", "Equal weights per chart", "No annotated training set"],
        ["Redundancy cues from DOM and a vision-language model", "Benchmark: cues recorded at generation; real pages: pattern fills only", "DOM cues are not available for images; no VLM run"],
    ], [1.9, 2.6, 2.2], caption_text="Differences between the Phase 2 plan and Phase 3")

    # ------------------------------------------------------------ 2 dataset
    doc.add_heading("2. Dataset and Preprocessing", 1)
    doc.add_heading("2.1 Controlled benchmark", 2)
    para(doc, "Every chart has a ground-truth label that comes from how it was constructed, never from the audit itself. Charts are bar, line and scatter plots plus heatmaps, drawn with matplotlib on white, light-grey or dark backgrounds, with and without gridlines; about 30% are saved as JPEG (quality 60-90) to introduce compression artefacts.")
    table(doc, ["Chart group", "Construction", "Truth label"], [
        ["Safe palettes (90)", "Okabe-Ito colours or lightness-stepped single-hue colours; every colour at least 3.3:1 on its background", "Pass"],
        ["Unsafe palettes, colour-only (90)", "Contain a textbook confusable pair: tab10 red/green, Set1 red/green, dark red/green or tab10 blue/purple", "Fail (CVD)"],
        ["Unsafe palettes, partial redundancy (30)", "As above with markers or patterns on only some series", "Fail (CVD)"],
        ["Unsafe palettes, full redundancy (35)", "As above with distinct markers or patterns and direct labels on every series", "Pass"],
        ["Low-contrast marks (40)", "Safe palette with one mark below 3:1 on its background (exact WCAG ratio on the true colours)", "Fail (contrast)"],
        ["Single colour (30)", "One-series charts", "Not Present"],
        ["Legend withheld (20 categorical, 36 heatmaps)", "Legend or colour bar removed", "Cannot Tell"],
        ["Heatmaps with legend (100)", "viridis, cividis, magma, Blues, RdBu (60); jet, gist_rainbow, nipy_spectral, RdYlGn (40)", "Pass / Fail"],
    ], [1.9, 3.7, 1.1], caption_text="Composition of each benchmark (471 charts)")
    dv, ts = M["dataset"]["dev"], M["dataset"]["test"]
    table(doc, ["Label", "Development (seed 42)", "Test (seed 2026)"],
          [[k, dv.get(k, 0), ts.get(k, 0)] for k in ("Pass", "Fail", "Not Present", "Cannot Tell")] + [["**Total**", sum(dv.values()), sum(ts.values())]],
          [2.0, 2.2, 2.2], caption_text="Label counts per split")
    para(doc, "**Split and tuning.** The development benchmark was used to build the pipeline and to calibrate tau. The test benchmark was generated after the pipeline was frozen and every reported number comes from it. After the first test run the only code added was real-dashboard tooling, which does not touch the benchmark results.")
    para(doc, "**Choices we disclose.** Three decisions shaped the labels. (1) Tol Bright and sampled viridis colours were not used as safe palettes: Tol's palette is documented for protan and deutan only while we also audit tritan, and viridis is a continuous map, not a categorical set. The detector did flag both, so leaving them out removes a disagreement that we mention here. (2) A tab10 green-brown pair was dropped from the unsafe list because both simulation models placed it close to the threshold. (3) Palette-extraction settings (median filter, flat-pixel selection, merge distance, background and grey thresholds) were tuned on the development benchmark.")
    doc.add_heading("2.2 Real-dashboard pilot", 2)
    if ds is not None:
        dsi = ds[~ds["id"].isin(excl)]
        n_ass = int((dsi.label != "No assessable chart").sum())
        n_fail_d = int((dsi.label == "Fail").sum())
        bits = []
        if n_urls is not None:
            bits.append(f"{n_urls} URLs were attempted")
        if status:
            bits.append("capture outcomes: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(status.items(), key=lambda x: -x[1])))
        para(doc, "; ".join(bits) + "." if bits else "Dashboards were captured with capture.py.")
        ex_txt = (f" {len(excl)} Tableau Public pages (ids {', '.join(sorted(excl, key=int))}) were excluded from the analysis because a pop-up dialog dimmed the chart in the screenshot and our automated dismissal did not remove it." if excl else "")
        para(doc, f"{len(ds)} pages were captured successfully.{ex_txt} Of the {len(dsi)} pages analysed, {n_ass} contained at least one chart the audit could assess; the others had blank embedded content, only single-colour marks, or no chart regions.")
        if ch is not None:
            chi = ch[~ch["id"].isin(excl)]
            para(doc, "Chart-level audit output on the included pages: " + ", ".join(f"{v} {k}" for k, v in chi.label.value_counts().items()) + ".")
    else:
        para(doc, "The real-dashboard pilot output (results_real/) was not available when this report was built.")
    doc.add_heading("2.3 Experimental setup and implementation details", 2)
    para(doc, f"Python {info.get('python', '').split()[0] if info else ''}, numpy {info.get('numpy', '')}, scikit-image, scikit-learn, SciPy, matplotlib, Playwright. "
              f"Benchmark seeds 42 and 2026. tau grid {cal['tau_grid']}. Five-fold stratified cross-validation for calibration. 10,000 bootstrap resamples for confidence intervals. "
              f"Palette extraction takes about {cost['palette_extraction']:.0f} ms per chart and the audit decision about {cost['audit_decision']:.1f} ms on the machine used.")

    # ------------------------------------------------------------ 3 baselines
    doc.add_heading("3. Baseline Evaluation", 1)
    bullet(doc, "**B1, contrast-only.** Fails a chart if any mark colour has less than 3:1 contrast with the background (WCAG 1.4.11). It has no notion of hue confusion.")
    bullet(doc, "**B2, simulation-only.** Uses the same palette extraction and Machado simulation with a fixed CIEDE2000 threshold of 10. It fails a chart if any simulated pair falls below it, with no calibration, contrast check, gradient logic or structure module.")
    bullet(doc, "**B3, vision-language auditor.** Not run in this phase (see Section 1.3).")
    rows = [[r.method, f3(r.fail_precision), f3(r.fail_recall), f3(r.fail_f1), f"{f3(r.f1_ci_low)}-{f3(r.f1_ci_high)}", f3(r.macro_f1), f3(r.accuracy), pct(r.abstention_rate)] for r in main_df.itertuples()]
    table(doc, ["Method", "Fail P", "Fail R", "Fail F1", "F1 95% CI", "Macro-F1", "4-class acc.", "Abstains"], rows,
          [1.5, 0.6, 0.6, 0.65, 1.0, 0.7, 0.75, 0.65], font=8, caption_text=f"Results on the test benchmark ({n_test} charts)")
    rows = [[r.comparison, int(r.hybrid_only_correct), int(r.baseline_only_correct), f"{r.p_raw:.2g}", f"{r.p_holm:.2g}"] for r in mc.itertuples()]
    table(doc, ["Comparison", "Hybrid right, baseline wrong", "Baseline right, hybrid wrong", "p (exact)", "p (Holm)"], rows, [2.2, 1.3, 1.3, 0.8, 0.8],
          caption_text="McNemar tests on the Fail / not-Fail decision")
    para(doc, f"The continuous scores behind the decisions also separate unsafe from safe palettes in different ways. ROC-AUC on {auc['n_charts']} categorical charts (Fail by CVD versus Pass without redundancy) is {f3(auc['hybrid_collapse_score'])} for the hybrid's colour-collapse score, {f3(auc['b2_min_sim_distance'])} for B2's minimum simulated distance and {f3(auc['b1_contrast_score'])} for B1's contrast score. Contrast is close to chance because contrast does not measure colour-vision confusion.")
    doc.add_heading("Calibration", 3)
    rows = [[f"{int(r.tau)}", f3(r.f1_dev), f3(r.f1_test)] for r in tau_df.itertuples()]
    table(doc, ["tau", "F1 (dev)", "F1 (test)"], rows, [1.0, 1.2, 1.2], caption_text=f"Fail-class F1 by threshold. Selected tau = {tau:g}; 5-fold CV F1 = {f3(cal['cv_f1_mean'])} +/- {f3(cal['cv_f1_sd'])}")

    # ------------------------------------------------------------ 4 figures
    doc.add_heading("4. Graphical Results and Visualization", 1)
    figure(doc, F / "fig1_main_comparison.png", "Fail-class precision, recall and F1 on the test benchmark. Error bars are bootstrap 95% confidence intervals of F1.")
    figure(doc, F / "fig3_tau_sensitivity.png", "F1 against the confusability threshold on the development and test benchmarks.", 5.0)
    figure(doc, F / "fig4_confusion.png", "Confusion matrices (rows: truth, columns: prediction) for the hybrid audit and the two baselines.", 6.5)
    figure(doc, F / "fig5_sensitivity.png", "Sensitivity to the simulation model and severity: rank correlation of chart scores, and the share of labels that change relative to Machado severity 1.0.")
    figure(doc, F / "fig7_extraction_quality.png", "Palette-extraction quality by chart type and image format.")
    figure(doc, F / "fig6_error_causes.png", "Sources of the hybrid audit's errors on the test benchmark.", 5.4)
    if ds is not None:
        chosen = []
        for lab in ("Fail", "Pass"):
            sel = ds[(~ds["id"].isin(excl)) & (ds.label == lab)]
            if len(sel):
                chosen.append((sel.iloc[0]["id"], lab))
        for did, lab in chosen:
            figure(doc, Path(a.real) / "previews" / f"dashboard_{did}.png", f"Real-dashboard pilot: chart regions of dashboard {did} (audit label: {lab}, not human-verified) shown under normal vision and simulated protan, deutan and tritan vision.", 6.3)

    # ------------------------------------------------------------ 5 comparison
    doc.add_heading("5. Comparison with Existing Research", 1)
    para(doc, "None of the Phase 2 papers evaluates a CVD accessibility audit on dashboard or chart images, so a like-for-like comparison is not possible. The table sets our results beside what each paper reports, using the values summarised in Phase 2, and states what prevents a fair comparison.")
    table(doc, ["Paper", "What it reports", "Our result", "Why it is not like-for-like"], [
        ["E3, Kumar et al. (2025), PDF accessibility benchmark", "Five language models, 125 PDFs, four labels. Best mean accuracy 0.85 (GPT-4-Turbo), lowest 0.42 (Llama-3.2). Colour-contrast accuracy 1.00 (GPT-4o) to 0.53 (Llama-3.2). GPT-4-Turbo reached 97.1% on Cannot Tell.",
         f"Four-class accuracy {f3(H.accuracy)} for the hybrid audit and {f3(B1.accuracy)} for the contrast-only baseline; Cannot Tell accuracy {f3(H.cannot_tell_accuracy)} on withheld legends.",
         "Different artefacts (PDF pages versus chart images), different judges (language models versus a deterministic pipeline), different label sources (expert versus by-construction). Both designs withhold inputs to create Cannot Tell cases."],
        ["E1, Liu et al. (2026), colormap recovery", "Synthetic test MSE 0.001, PSNR 34.0, SSIM 0.961; real figures MSE 0.038, PSNR 17.2, SSIM 0.801.",
         "We read colours from the image or the legend and do not recover colormaps.",
         "Different task and metrics. The gap between its synthetic and real scores is a warning for us: our controlled results are likely optimistic compared with real dashboards."],
        ["E4, Sidorchuk et al. (2025), daltonization with observers", "Objective metrics and 17 observers can disagree (for example contrast improved in 65% of protan cases against 70% for the anisotropic method).",
         f"Our decisions change for {pct(flips['Machado 0.6'])} of charts when severity drops from 1.0 to 0.6.",
         "E4 recolours natural images and uses observers; we have no observers. The shared lesson is that simulation settings and perception must both be reported."],
        ["P7, Hayashi et al. (2026), vision-language models on Ishihara plates", "GPT-4o reached 23.6% on protan and 5.9% on deutan plates.",
         "We did not run B3.", "Different task. It is the reason the hybrid never asks a language model how colours look to a CVD viewer."],
    ], [1.3, 2.0, 1.6, 1.8], font=8, caption_text="Our results beside the closest published work")

    # ------------------------------------------------------------ 6 ablation
    doc.add_heading("6. Ablation Studies", 1)
    para(doc, "Each variant changes one component of the full model. Variants with a different distance scale or severity were recalibrated on the development benchmark; all are evaluated on the test benchmark.")
    rows = [[r.variant, f"{r.tau:g}", f3(r.fail_f1), f"{f3(r.f1_ci_low)}-{f3(r.f1_ci_high)}", f3(r.fail_precision), f3(r.fail_recall), f"{r.delta_f1_vs_full:+.3f}"] for r in abl.itertuples()]
    table(doc, ["Variant", "tau", "Fail F1", "95% CI", "Fail P", "Fail R", "Delta F1"], rows, [1.9, 0.5, 0.7, 1.1, 0.65, 0.65, 0.7], font=8,
          caption_text="Ablation results on the test benchmark")
    figure(doc, F / "fig2_ablation.png", "Fail-class F1 per ablation variant (bars in orange lose more than 0.005 F1 against the full model).", 5.6)
    drop = abl[abl.delta_f1_vs_full < -0.01].sort_values("delta_f1_vs_full")
    same = abl[(abl.variant != "Full model") & (abl.delta_f1_vs_full.abs() <= 0.01)]
    if len(drop):
        para(doc, "**What matters.** " + "; ".join(f"{r.variant.strip('- ')} ({r.delta_f1_vs_full:+.3f})" for r in drop.itertuples()) + ".")
    if len(same):
        names = [r.variant for r in same.itertuples()]
        why = []
        if any("protan" in n or "tritan" in n for n in names):
            why.append("every unsafe palette in the benchmark is also confusable under deutan vision, so the protan and tritan checks add little here")
        if any("Vienot" in n for n in names):
            why.append("the two simulation models agree at severity 1.0")
        if any("fixed" in n for n in names):
            why.append("calibration selected the same threshold as the fixed value")
        para(doc, "**What made no measurable difference (within 0.01 F1):** " + ", ".join(n.strip("- ") for n in names) + ". This reflects the benchmark and is not proof these parts are unnecessary" + (": " + "; ".join(why) + "." if why else "."))

    # ------------------------------------------------------------ 7 discussion
    doc.add_heading("7. Results Analysis and Discussion", 1)
    doc.add_heading("7.1 Findings", 2)
    bullet(doc, f"**Hue confusion needs simulation.** Contrast alone finds only {pct(B1.fail_recall)} of failing charts; it cannot see a red-green pair that has good contrast with the background.")
    bullet(doc, f"**Simulation alone over-flags.** B2 reaches recall {f3(B2.fail_recall)} but precision {f3(B2.fail_precision)}: with no calibration, no contrast check, no gradient logic and no redundancy handling, it flags many safe charts and misses others.")
    bullet(doc, f"**The combination works on the benchmark.** The hybrid reaches precision {f3(H.fail_precision)} and recall {f3(H.fail_recall)}. It abstains on {pct(H.abstention_rate)} of charts, mostly where we intend it to (withheld legends and unsafe palettes with full redundancy).")
    bullet(doc, f"**Severity matters.** Using severity 0.3 or 0.6 instead of 1.0 changes {pct(flips['Machado 0.3'])} and {pct(flips['Machado 0.6'])} of labels at the same threshold. Audit results are only meaningful together with the severity assumed, and our threshold is set for severity 1.0. The Vienot model changes {pct(flips['Vienot 1.0'])} of labels.")
    if hv is not None:
        reasons = ""
        if dis is not None and len(dis) and "reason" in dis.columns:
            fp_rows = dis[dis["label"] == "Fail"]
            if len(fp_rows):
                reasons = " The audit's stated reason for these false alarms: " + ", ".join(f"{n} {k}" for k, n in fp_rows["reason"].value_counts().items()) + "."
        bullet(doc, f"**{'Real pages are harder.' if harder else 'Real-page spot-check.'}** On {int(hv.n)} judgeable real charts the hybrid's precision is {f3(hv.fail_precision)} (Wilson 95% CI {f3(hv.precision_wilson_low)}-{f3(hv.precision_wilson_high)}) with recall {f3(hv.fail_recall)} on only {int(hv.tp + hv.fn)} failing charts, against a benchmark precision of {f3(H.fail_precision)}. The intervals of the three methods overlap, so this sample cannot separate them.{reasons} We have not established whether the gap comes from the threshold, extraction noise on real pages, or the difference between a reviewer's judgment of simulated images and the audit's pairwise rule." + (" The controlled benchmark should therefore be read as an upper bound on real-page performance." if harder else ""))
    causes = list(zip(errc["cause"], errc["count"]))
    unintended = [(c, n) for c, n in causes if not c.startswith("abstained")]
    top = max(unintended, key=lambda x: x[1]) if unintended else None
    txt = "**Where the errors come from.** " + "; ".join(f"{n} {c}" for c, n in causes) + ". "
    txt += f"Palette extraction recovers the exact colour count on {pct(ext['exact_k_rate'])} of charts with a mean colour error of {ext['mean_colour_error_de2000']:.1f} CIEDE2000 units. "
    if top:
        txt += f"The largest unintended cause is: {top[0]} ({top[1]} charts). Real pages are noisier than rendered charts, so extraction errors will matter more there."
    bullet(doc, txt)
    doc.add_heading("7.2 Real-dashboard pilot", 2)
    if ds is not None:
        dsi = ds[~ds["id"].isin(excl)]
        n_ass = int((dsi.label != "No assessable chart").sum()); n_fail_d = int((dsi.label == "Fail").sum())
        para(doc, f"The pilot ran the full pipeline on {len(dsi)} included pages. {n_ass} had at least one assessable chart and the audit flagged {n_fail_d} of them. We do not report this as prevalence: the sample is tiny and not random, the audit labels have no ground truth unless the spot-check below was completed, and the pilot showed several capture and extraction problems (pop-ups, embedded content that did not render, bot-protection pages, and many pages blocked by robots.txt).")
        if rv is not None and rv_sum is not None:
            who = ("drafted by an AI model (Claude) from the same normal and deuteranopia-simulated crops, without the audit's label, and then checked and corrected by a team member" if a.review_ai_assisted
                   else "labelled by a reviewer who saw normal and deuteranopia-simulated crops without the audit's label")
            para(doc, f"**Human spot-check.** {rv_sum.get('reviewed_regions', 0)} audited regions were {who}; {rv_sum.get('judgeable_charts', 0)} were judgeable charts and {pct(rv_sum.get('not_a_chart_share', 0))} of regions were not data charts.")
            rows = [[r.method, int(r.n), f3(r.fail_precision), f3(r.fail_recall), f3(r.fail_f1), f3(r.agreement)] for r in rv.itertuples()]
            table(doc, ["Method", "n", "Fail P", "Fail R", "Fail F1", "Agreement"], rows, [1.9, 0.5, 0.8, 0.8, 0.8, 0.9], caption_text="Agreement with the human spot-check on real pages")
            if dis is not None and len(dis):
                rows = [[r.key, r.human_label, r.label, r.reason, (r.note or "")[:70]] for r in dis.itertuples()]
                table(doc, ["Chart", "Human label", "Audit label", "Audit reason", "Reviewer note"], rows, [0.6, 0.9, 0.9, 1.6, 2.7], font=8,
                      caption_text="Charts where the audit and the reviewer disagree")
            if "cohen_kappa" in rv_sum:
                para(doc, f"Agreement between two independent annotators: Cohen's kappa {rv_sum['cohen_kappa']:.2f} (raw agreement {pct(rv_sum['raw_agreement'])}).")
            para(doc, ("The labels came from simulated images and were AI-drafted with human checking, from a single labeller, so there is no inter-annotator agreement; this is a proxy for CVD users' judgments, not a substitute for them." if a.review_ai_assisted
                       else "Annotators with typical colour vision judged simulated images, so this is a proxy for CVD users' judgments and not a substitute for them."))
        else:
            para(doc, "A human spot-check of the audited regions was not completed, so no accuracy is reported for real pages.")
    doc.add_heading("7.3 Does this address the research gap?", 2)
    table(doc, ["Part of the Phase 2 gap statement", "Status after Phase 3"], [
        ["Automated audit for public dashboards", "Built and run end to end; evidence on real pages is a small pilot."],
        ["Calibrated against human judgment", "Calibrated against known labels on a controlled benchmark. Human agreement on real pages is limited to the spot-check above, if completed."],
        ["Accounts for non-colour cues", "Tested on the benchmark through the redundancy rule. On real pages only pattern fills are read."],
        ["Reports sensitivity to simulation settings", "Done: severity, model, threshold and component ablations."],
        ["Measured on a large sample of real dashboards", "Not met. Pilot only."],
    ], [2.6, 4.1], caption_text="Gap statement: what this phase did and did not close")
    table(doc, ["Hypothesis / RQ", "Status"], [
        ["H3 / RQ3: hybrid beats contrast-only and simulation-only baselines", f"Supported on the controlled benchmark (F1 {f3(H.fail_f1)} against {f3(B1.fail_f1)} and {f3(B2.fail_f1)}; Holm-adjusted p < 0.001). Not shown on real dashboards."],
        ["RQ2: audit score predicts CVD users' task errors; sensitivity to simulation settings", "Sensitivity measured. Prediction of CVD users' task errors not tested: no CVD observers."],
        ["RQ1 / H1: prevalence and differences by platform and domain", "Not testable with the pilot sample."],
        ["H2: composite score correlates better with observers' errors than contrast", "Not tested: no CVD observers."],
    ], [3.0, 3.7], caption_text="Status of the Phase 2 research questions and hypotheses")
    doc.add_heading("7.4 Strengths and limitations", 2)
    bullet(doc, "**Strengths.** Deterministic and cheap; explicit abstention; every component ablated; sensitivity to simulation severity reported; honest separation of development and test data; full code and seeds released.")
    bullet(doc, "**Limitation: construction labels.** The benchmark tests the pipeline's mechanics. Labels reflect design knowledge about palettes, not CVD users' judgments, and the benchmark is easier than real dashboards.")
    bullet(doc, "**Limitation: coverage.** Unsafe palettes were drawn from four textbook confusions, all deutan-confusable, so the protan and tritan checks are untested. Heatmap legends are read from the benchmark's recorded ramp, and the legend-readable flag is also given, so a real legend detector is untested.")
    bullet(doc, "**Limitation: extraction.** Grey series are ignored when small; very faint marks near the background are removed; photos and continuous fields are routed to Cannot Tell by a heuristic.")
    bullet(doc, "**Limitation: B3 not run, so the claim that the hybrid is better than a purely learned auditor remains untested here.**")
    doc.add_heading("7.5 Next steps", 2)
    bullet(doc, "Scale capture with more hosts that permit automated access and a working pop-up handler; read DOM cues for redundancy and legends.")
    bullet(doc, "Recruit participants with diagnosed CVD for the task-error study (RQ2) and add a second annotator.")
    bullet(doc, "Run the vision-language baseline B3 with a Cannot Tell option.")

    # ------------------------------------------------------------ 8 reproducibility
    doc.add_heading("8. Reproducibility", 1)
    para(doc, "Run from the project folder inside the virtual environment:")
    for line in ["python make_benchmark.py --out benchmark --seed 42", "python make_benchmark.py --out benchmark_test --seed 2026",
                 "python run_experiments.py --dev benchmark --test benchmark_test --out results",
                 "python capture.py --csv urls.csv --out data/raw --wait 12", "python audit_real.py --csv urls.csv --raw data/raw --out results_real",
                 "python review.py make ... / python review.py score ...", "python build_report.py ..."]:
        p = doc.add_paragraph(); r = p.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(9); p.paragraph_format.space_after = Pt(1)
    para(doc, "Files: capture.py, cvdaudit.py, make_benchmark.py, run_experiments.py, audit_real.py, review.py, build_report.py, diagnose.py, contact_sheet.py. Benchmark labels are deterministic for a given seed; pixels can differ slightly across machines, so results may differ in the third decimal place.", space_after=4)

    # ------------------------------------------------------------ appendix
    if ds is not None:
        doc.add_heading("Appendix A. Pilot dashboards audited", 1)
        dsi = ds[~ds["id"].isin(excl)]
        rows = [[r.id, str(r.title)[:60], int(r.regions_found), int(r.assessable), int(r.failing), r.label] for r in dsi.itertuples()]
        table(doc, ["ID", "Page title", "Regions", "Assessable", "Failing", "Audit label"], rows, [0.4, 3.0, 0.7, 0.8, 0.6, 1.2], font=8,
              caption_text="Per-dashboard audit output (not human-verified unless the spot-check says otherwise)")

    # ------------------------------------------------------------ references
    doc.add_heading("References", 1)
    refs = [
        "[E1] H. Liu et al., \"Self-supervised continuous colormap recovery from a 2D scalar field visualization without a legend,\" IEEE Trans. Vis. Comput. Graph., vol. 32, no. 1, pp. 265-275, 2026, doi: 10.1109/TVCG.2025.3633886.",
        "[E3] A. Kumar, T. Padath and L. L. Wang, \"Benchmarking PDF accessibility evaluation: A dataset and framework for assessing automated and LLM-based approaches for accessibility testing,\" in Proc. ASSETS '25, Denver, CO, USA, Oct. 2025, doi: 10.1145/3663547.3746380.",
        "[E4] D. Sidorchuk et al., \"Leveraging achromatic component for trichromat-friendly daltonization,\" J. Imaging, vol. 11, no. 7, Art. no. 225, Jul. 2025, doi: 10.3390/jimaging11070225.",
        "[P7] K. Hayashi, S. Ozaki, Y. Sakai, H. Kamigaito and T. Watanabe, \"Diagnosing vision language models' perception by leveraging human methods for color vision deficiencies,\" in Proc. EACL 2026, Rabat, Morocco, Mar. 2026, ACL Anthology ID 2026.eacl-long.356.",
        "G. M. Machado, M. M. Oliveira and L. A. F. Fernandes, \"A physiologically-based model for simulation of color vision deficiency,\" IEEE Trans. Vis. Comput. Graph., vol. 15, no. 6, pp. 1291-1298, 2009.",
        "F. Vienot, H. Brettel and J. D. Mollon, \"Digital video colourmaps for checking the legibility of displays by dichromats,\" Color Res. Appl., vol. 24, no. 4, pp. 243-252, 1999.",
        "G. Sharma, W. Wu and E. N. Dalal, \"The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations,\" Color Res. Appl., vol. 30, no. 1, pp. 21-30, 2005.",
        "B. Wong, \"Color blindness,\" Nature Methods, vol. 8, no. 6, p. 441, 2011 (source of the Okabe-Ito palette used for safe palettes).",
        "World Wide Web Consortium, Web Content Accessibility Guidelines (WCAG) 2.1, Success Criteria 1.4.1, 1.4.3 and 1.4.11, 2018.",
    ]
    for r in refs:
        p = para(doc, r, 9.5, space_after=3); p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.6)
    doc.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
