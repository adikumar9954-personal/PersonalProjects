import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")  # handle accented player names on Windows

from output.report import run_full_pipeline
from output.html_report import generate_report

season = "2025-26"
out = run_full_pipeline(season=season, max_games=500)

report_path = Path(__file__).parent / "reports" / f"report_{season}.html"
generate_report(out, path=str(report_path), season=season)
print(f"\nReport ready: {report_path}")
