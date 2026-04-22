import sys
sys.path.insert(0, r"C:\nba_valuation")

from output.team_report import generate_team_report

# Hornets — with date split to capture the injury-return turnaround
generate_team_report(
    team="hornets",
    season="2025-26",
    split_date="2025-12-15",    # adjust to when the winning run started
    path=r"C:\nba_valuation\output\hornets_report.html",
)

# Lakers
generate_team_report(
    team="lakers",
    season="2025-26",
    split_date="2025-12-15",
    path=r"C:\nba_valuation\output\lakers_report.html",
)