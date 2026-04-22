from nba_api.stats.endpoints import leaguehustlestatsplayer, leaguedashplayerptshot
import inspect, time

print("=== hustle params ===")
for name, param in inspect.signature(leaguehustlestatsplayer.LeagueHustleStatsPlayer.__init__).parameters.items():
    if name != "self": print(f"  {name} = {param.default!r}")

print("\n=== hustle columns ===")
ep = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
    season="2023-24",
    season_type_all_star="Regular Season",
)
df = ep.get_data_frames()[0]
print(df.columns.tolist())
print(len(df))

time.sleep(4)

print("\n=== ptshot columns ===")
ep2 = leaguedashplayerptshot.LeagueDashPlayerPtShot(
    season="2023-24",
    season_type_all_star="Regular Season",
)
df2 = ep2.get_data_frames()[0]
print(df2.columns.tolist())
print(len(df2))