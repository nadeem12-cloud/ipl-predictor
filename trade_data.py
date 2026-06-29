"""
trade_data.py
─────────────
Trade Predictor data layer.

Two independent pieces:
  1. Team Needs Analyzer — rules-based, uses squad role counts already
     defined in app.py (TEAM_SQUADS + PLAYER_ROLES). No external data needed.
  2. Rumor Mill — pulls real headlines from ESPN Cricinfo's public RSS feed
     and filters for IPL team names + trade-related keywords. No API key.

Everything here is explainable and honest — there is no ML model
"predicting" trades. Trades are rare, human-negotiated events with no
clean historical label set, so this is a needs-matching + news-surfacing
tool instead of a fake prediction model.
"""

import streamlit as st
import xml.etree.ElementTree as ET
import re
from urllib.request import Request, urlopen

CRICINFO_RSS_URL    = "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"
CRICTRACKER_RSS_URL  = "https://www.crictracker.com/feed/"
RSS_FEEDS = [CRICINFO_RSS_URL, CRICTRACKER_RSS_URL]

# Ideal role distribution for a balanced 11 (used as the needs baseline)
IDEAL_ROLE_RATIO = {"BAT": 4, "BOWL": 4, "ALL": 2, "WK": 1}

TRADE_KEYWORDS = [
    "trade", "traded", "trades", "swap", "swapped", "transfer",
    "transferred", "released", "retain", "retention", "auction",
    "sign", "signs", "signed", "move to", "moves to", "joins",
    "deal", "exchange"
]


# ── TEAM NEEDS ANALYZER (rules-based, no external data) ──────────────────────

def analyze_team_needs(team_name: str, squad: list, player_roles: dict) -> dict:
    """
    Given a team's current squad, compute role surplus/gap vs an ideal
    balanced XI composition (4 BAT, 4 BOWL, 2 ALL, 1 WK as a baseline ratio,
    scaled to squad size).

    Returns:
    {
        "counts": {"BAT": n, "BOWL": n, "ALL": n, "WK": n},
        "gaps":    [role, role, ...]   # roles below ideal proportion
        "surplus": [role, role, ...]   # roles above ideal proportion
        "balanced": bool
    }
    """
    counts = {"BAT": 0, "BOWL": 0, "ALL": 0, "WK": 0}
    for player in squad:
        role = player_roles.get(player, "BAT")
        counts[role] = counts.get(role, 0) + 1

    squad_size = len(squad) or 1
    ideal_total = sum(IDEAL_ROLE_RATIO.values())

    gaps, surplus = [], []
    for role, ideal_count in IDEAL_ROLE_RATIO.items():
        ideal_proportion = ideal_count / ideal_total
        actual_proportion = counts.get(role, 0) / squad_size

        # Allow a tolerance band before flagging as gap/surplus
        if actual_proportion < ideal_proportion - 0.05:
            gaps.append(role)
        elif actual_proportion > ideal_proportion + 0.08:
            surplus.append(role)

    return {
        "counts": counts,
        "gaps": gaps,
        "surplus": surplus,
        "balanced": not gaps and not surplus,
    }


def compute_trade_compatibility(team_a: str, team_b: str, needs_a: dict, needs_b: dict) -> dict:
    """
    Compares two teams' needs/surplus profiles and returns a compatibility
    score + suggested role-based swap directions.

    This is NOT predicting a specific trade — it's showing where a
    structurally sensible swap *could* make sense based on squad balance.
    """
    a_gaps, a_surplus = set(needs_a["gaps"]), set(needs_a["surplus"])
    b_gaps, b_surplus = set(needs_b["gaps"]), set(needs_b["surplus"])

    # A's surplus filling B's gap
    a_to_b = a_surplus & b_gaps
    # B's surplus filling A's gap
    b_to_a = b_surplus & a_gaps

    score = len(a_to_b) + len(b_to_a)
    max_score = 4  # at most all 4 roles align both ways (theoretical ceiling)

    return {
        "score": score,
        "score_pct": round(score / max_score * 100),
        "a_to_b": sorted(a_to_b),   # roles team_a has spare that team_b needs
        "b_to_a": sorted(b_to_a),   # roles team_b has spare that team_a needs
        "mutual": score >= 2,
    }


def role_display(role: str) -> str:
    return {"BAT": "Batters", "BOWL": "Bowlers", "ALL": "All-rounders", "WK": "Wicketkeepers"}.get(role, role)


# ── RUMOR MILL (live RSS, no API key) ─────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trade_rumors(team_names: list, max_items: int = 12) -> list:
    """
    Fetches cricket news RSS feeds (tries ESPN Cricinfo first, then
    CricTracker as a fallback) and filters headlines that mention both
    a trade-related keyword AND an IPL team/player context.

    Returns list of dicts: { title, description, link, pub_date, source }
    Falls back to [] on total failure — caller should handle empty state.
    """
    team_names_lower = [t.lower() for t in team_names]
    short_forms = ["ipl", "mumbai indians", "chennai super kings",
                   "royal challengers", "kolkata knight riders",
                   "sunrisers hyderabad", "rajasthan royals",
                   "gujarat titans", "punjab kings", "delhi capitals",
                   "lucknow super giants"]

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for feed_url in RSS_FEEDS:
        try:
            req = Request(feed_url, headers=headers)
            with urlopen(req, timeout=8) as resp:
                raw = resp.read()

            root = ET.fromstring(raw)
            items = root.findall(".//item")
            source_name = "ESPN Cricinfo" if "espncricinfo" in feed_url else "CricTracker"

            results = []
            for item in items:
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "").strip()
                desc  = re.sub(r"<[^>]+>", "", desc)  # strip any embedded HTML
                link  = (item.findtext("link") or item.findtext("url") or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()

                haystack = f"{title} {desc}".lower()
                has_keyword = any(kw in haystack for kw in TRADE_KEYWORDS)
                has_ipl_context = any(t in haystack for t in short_forms + team_names_lower)

                if has_keyword and has_ipl_context:
                    results.append({
                        "title": title,
                        "description": desc,
                        "link": link,
                        "pub_date": pub,
                        "source": source_name,
                    })

                if len(results) >= max_items:
                    break

            if results:
                return results
            # If this feed returned 0 matches, fall through and try next feed

        except Exception:
            continue  # try next feed in RSS_FEEDS

    return []


def clean_rss_link(url: str) -> str:
    """Strip tracking params like ?ex_cid=OTC-RSS for a cleaner display link."""
    return re.sub(r"\?.*$", "", url)