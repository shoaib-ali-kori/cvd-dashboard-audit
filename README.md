# Automated Auditing of Color-Vision Accessibility in Public Dashboards - Phase 3

DAV Research Project, FAST-NUCES Islamabad. Code, results and report for Phase 3.

## Setup
    python -m venv venv
    venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m playwright install chromium     (only needed for capture.py)

## Run order
1. Controlled benchmark and experiments (no browser needed)
    python make_benchmark.py --out benchmark --seed 42
    python make_benchmark.py --out benchmark_test --seed 2026
    python run_experiments.py --dev benchmark --test benchmark_test --out results
   Writes results/metrics.json, results/tables/*.csv, results/figures/*.png.
2. Real-dashboard pilot
    python capture.py --csv urls.csv --out data/raw --wait 12
    python diagnose.py --csv urls.csv --out data/raw
    python audit_real.py --csv urls.csv --raw data/raw --out results_real
3. Human spot-check
    python review.py make --charts results_real/charts.csv --raw data/raw --out review --exclude 1,2,3,4,5
    (fill review/review_labels.csv; edit it in Notepad or VS Code, not Excel)
    python review.py score --labels review/review_labels.csv --charts results_real/charts.csv --out review
4. Report
    python build_report.py --results results --real results_real --raw data/raw --urls urls.csv --exclude-ids 1,2,3,4,5 --review-ai-assisted --out DAV_Phase3_Implementation_and_Evaluation.docx

## Files
- cvdaudit.py: audit library (CVD simulation, palette extraction, features, hybrid audit, baselines B1 and B2)
- make_benchmark.py: synthetic benchmark with ground-truth labels by construction
- run_experiments.py: calibration, baselines, ablations, statistics, sensitivity, figures
- capture.py, diagnose.py, contact_sheet.py: dashboard capture and inspection
- audit_real.py, review.py: real-page audit and human spot-check
- build_report.py: builds the Word report from the results folders
- urls.csv: dashboards attempted (URLs only; screenshots are not redistributed)

## Notes
- Development benchmark seed 42; test benchmark seed 2026, used for all reported numbers.
- Benchmark labels are deterministic for a given seed; pixels can differ slightly across machines.
- The spot-check labels in review/ were drafted by an AI model from the review sheets and checked by a team member (see review_labels_draft.csv).
- B3 (vision-language baseline) was not run.
