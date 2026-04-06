"""
cricinfo_data.py
────────────────
Live IPL 2026 data via the `cricdata` package (MIT licensed).
Scrapes __NEXT_DATA__ JSON from ESPN Cricinfo public pages.

Data provided:
  - Points table (standings)
  - Match results → team form + H2H
  - Player career T20 stats
  - Current season stats

All functions are cached with st.cache_data (TTL 30 min).
Falls back gracefully if Cricinfo is unreachable.
"""

import streamlit as st
from cricdata import CricinfoClient

# IPL 2026 series slug — from espncricinfo.com/series/ipl-2026-1510719
IPL_SLUG = "ipl-2026-1510719"

TEAM_NAME_MAP = {
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":  "Royal Challengers Bengaluru",
    "Mumbai Indians":               "Mumbai Indians",
    "Chennai Super Kings":          "Chennai Super Kings",
    "Kolkata Knight Riders":        "Kolkata Knight Riders",
    "Sunrisers Hyderabad":          "Sunrisers Hyderabad",
    "Rajasthan Royals":             "Rajasthan Royals",
    "Gujarat Titans":               "Gujarat Titans",
    "Punjab Kings":                 "Punjab Kings",
    "Delhi Capitals":               "Delhi Capitals",
    "Lucknow Super Giants":         "Lucknow Super Giants",
}

ALL_TEAMS = list(set(TEAM_NAME_MAP.values()))


def _norm(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ── POINTS TABLE ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_points_table() -> list:
    """
    Returns list of dicts — one per team:
    { team, short, played, won, lost, nr, pts, nrr }
    Sorted by points descending.
    Falls back to [] on error.
    """
    try:
        ci   = CricinfoClient()
        data = ci.series_standings(IPL_SLUG)

        # Navigate the nested structure
        groups = (
            data.get("content", {})
                .get("standings", {})
                .get("groups", [])
        )
        rows = []
        for grp in groups:
            for team in grp.get("teamStats", []):
                info  = team.get("teamInfo", {})
                tname = _norm(info.get("longName", ""))
                rows.append({
                    "team":   tname,
                    "short":  info.get("abbreviation", tname[:3].upper()),
                    "played": int(team.get("matchesPlayed", 0)),
                    "won":    int(team.get("matchesWon", 0)),
                    "lost":   int(team.get("matchesLost", 0)),
                    "nr":     int(team.get("matchesNoResult", 0)),
                    "pts":    int(team.get("points", 0)),
                    "nrr":    float(team.get("netRunRate", 0.0)),
                })

        rows.sort(key=lambda x: (x["pts"], x["nrr"]), reverse=True)
        return rows

    except Exception as e:
        st.warning(f"⚠️ Could not fetch points table: {e}", icon="🌐")
        return []


# ── MATCH RESULTS ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_match_results() -> list:
    """
    Returns list of completed match dicts (sorted oldest-first):
    { date, team1, team2, winner, margin, venue }
    Falls back to [] on error.
    """
    try:
        ci   = CricinfoClient()
        data = ci.series_matches(IPL_SLUG)

        matches_raw = (
            data.get("content", {})
                .get("matches", [])
        )

        results = []
        for m in matches_raw:
            # Skip if not completed
            status = m.get("matchInfo", {}).get("state", "")
            if status.lower() not in ("complete", "result"):
                continue

            date  = m.get("matchInfo", {}).get("startDate", "")[:10]
            venue = m.get("matchInfo", {}).get("ground", {}).get("longName", "")

            teams = m.get("teams", [])
            if len(teams) < 2:
                continue

            team1 = _norm(teams[0].get("team", {}).get("longName", ""))
            team2 = _norm(teams[1].get("team", {}).get("longName", ""))

            # Winner
            winner = ""
            result_str = m.get("matchInfo", {}).get("status", "")
            for t in teams:
                if t.get("isWinner"):
                    winner = _norm(t.get("team", {}).get("longName", ""))
                    break
            if not winner and "no result" in result_str.lower():
                winner = "No Result"

            # Margin — usually in status string e.g. "MI won by 6 wickets"
            margin = result_str

            results.append({
                "date":   date,
                "team1":  team1,
                "team2":  team2,
                "winner": winner,
                "margin": margin,
                "venue":  venue,
            })

        results.sort(key=lambda x: x["date"])
        return results

    except Exception as e:
        st.warning(f"⚠️ Could not fetch match results: {e}", icon="🌐")
        return []


# ── DERIVED: FORM + H2H ───────────────────────────────────────────────────────

def compute_team_form(results: list, team: str, window: int = 5) -> tuple:
    """
    Returns (form_float, last5_list) from match results.
    form_float: win rate in last `window` matches (0.0–1.0)
    last5_list: list of "W" / "L" / "NR" strings
    """
    team_matches = [
        m for m in results
        if team in (m["team1"], m["team2"])
    ]
    last = team_matches[-window:] if len(team_matches) >= window else team_matches

    wl = []
    for m in last:
        if m["winner"] == "No Result":
            wl.append("NR")
        elif m["winner"] == team:
            wl.append("W")
        else:
            wl.append("L")

    wins = wl.count("W")
    rate = wins / len(wl) if wl else 0.5
    return round(rate, 4), wl


def compute_h2h(results: list, team1: str, team2: str, window: int = 10) -> tuple:
    """
    Returns (meetings_list, t1_win_rate).
    meetings_list: [(winner, date, margin, venue), ...]
    """
    meetings = [
        (m["winner"], m["date"], m["margin"], m["venue"])
        for m in results
        if set([m["team1"], m["team2"]]) == set([team1, team2])
    ]
    last     = meetings[-window:]
    t1_wins  = sum(1 for w, _, _, _ in last if w == team1)
    rate     = t1_wins / len(last) if last else 0.5
    return last, round(rate, 4)


# ── PLAYER STATS ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_id(name: str) -> int | None:
    """Search Cricinfo for player ID by name. Cached 1 hour."""
    try:
        ci      = CricinfoClient()
        results = ci.search_players(name, limit=3)
        if results:
            return results[0].get("id")
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_stats(player_id: int) -> dict:
    """
    Returns { batting: {...}, bowling: {...} } for T20 career.
    Falls back to {} on error.
    """
    if not player_id:
        return {}
    try:
        ci = CricinfoClient()

        bat_data  = ci.player_career_stats(player_id, fmt="t20", stat_type="batting")
        bowl_data = ci.player_career_stats(player_id, fmt="t20", stat_type="bowling")

        bat_summary  = bat_data.get("summary", {})
        bowl_summary = bowl_data.get("summary", {})

        return {
            "batting": {
                "mat":  bat_summary.get("Mat", "—"),
                "runs": bat_summary.get("Runs", "—"),
                "avg":  bat_summary.get("Ave", "—"),
                "sr":   bat_summary.get("SR", "—"),
                "50s":  bat_summary.get("50", "—"),
                "100s": bat_summary.get("100", "—"),
            } if bat_summary else {},
            "bowling": {
                "mat":  bowl_summary.get("Mat", "—"),
                "wkts": bowl_summary.get("Wkts", "—"),
                "avg":  bowl_summary.get("Ave", "—"),
                "eco":  bowl_summary.get("Econ", "—"),
                "best": bowl_summary.get("BBI", "—"),
            } if bowl_summary else {},
        }
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_season_stats() -> dict:
    """
    Fetch IPL 2026 season batting + bowling top stats.
    Returns { batting: [...], bowling: [...] }
    """
    try:
        ci   = CricinfoClient()
        data = ci.series_stats(IPL_SLUG)

        content = data.get("content", {})
        batting = content.get("topBatsmen", [])
        bowling = content.get("topBowlers", [])

        def parse_bat(p):
            return {
                "name":  p.get("player", {}).get("longName", ""),
                "team":  _norm(p.get("team", {}).get("longName", "")),
                "runs":  p.get("runs", 0),
                "avg":   p.get("average", 0),
                "sr":    p.get("strikeRate", 0),
                "50s":   p.get("fifties", 0),
            }

        def parse_bowl(p):
            return {
                "name":  p.get("player", {}).get("longName", ""),
                "team":  _norm(p.get("team", {}).get("longName", "")),
                "wkts":  p.get("wickets", 0),
                "avg":   p.get("average", 0),
                "eco":   p.get("economy", 0),
                "best":  p.get("bestBowling", "—"),
            }

        return {
            "batting": [parse_bat(p) for p in batting[:10]],
            "bowling": [parse_bowl(p) for p in bowling[:10]],
        }
    except Exception:
        return {"batting": [], "bowling": []}