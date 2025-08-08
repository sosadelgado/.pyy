
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import time
import re

app = FastAPI(title="Bolt Backend", description="Bolt: CS2 prop evaluator backend (HLTV pure-scrape cache)")

# Simple in-memory cache with TTL
_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 30 * 60  # 30 minutes

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BoltBot/1.0"
}

# --- Pydantic models ---
class EvaluateRequest(BaseModel):
    player: str
    kill_line: Optional[float] = 0.0
    hs_line: Optional[float] = 0.0
    salary: Optional[float] = 0.0
    map_count: Optional[int] = 2
    kpr: Optional[float] = None
    hs_percent: Optional[float] = None

# --- Utilities ---
def cache_get(key: str):
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        try:
            del _cache[key]
        except KeyError:
            pass
        return None
    return entry["value"]

def cache_set(key: str, value):
    _cache[key] = {"ts": time.time(), "value": value}

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def parse_player_stats_from_player_page(html: str):
    \"\"\"Return (kpr, hs_percent) where hs_percent is a decimal (0.42)\"\"\"
    soup = BeautifulSoup(html, \"html.parser\")
    kpr = None
    hs = None

    # Look for stats blocks that contain 'Kills / round' and 'Headshots'
    try:
        # Many HLTV pages have <span class="statsLabel">Kills / round</span><span class="statsValue">0.72</span>
        labels = soup.find_all(lambda tag: tag.name in [\"span\",\"div\"] and tag.get_text(strip=True) in [\"Kills / round\", \"Headshots\"])
        if labels:
            for lab in labels:
                text = lab.get_text(strip=True)
                sibling = lab.find_next_sibling()
                if sibling:
                    val = sibling.get_text(strip=True).replace('%','').strip()
                    if text == \"Kills / round\":
                        kpr = safe_float(val)
                    elif text == \"Headshots\":
                        hs = safe_float(val)
                        if hs is not None and hs > 1:
                            hs = hs / 100.0
        # Fallback: regex search for 'Kills / round' pattern in page
        if kpr is None:
            m = re.search(r'Kills\\s*/\\s*round[^0-9]*([0-9]+\\.?[0-9]*)', html, re.IGNORECASE)
            if m:
                kpr = safe_float(m.group(1))
        if hs is None:
            m = re.search(r'Headshots[^0-9%]*([0-9]+\\.?[0-9]*)%?', html, re.IGNORECASE)
            if m:
                val = safe_float(m.group(1))
                if val is not None and val > 1:
                    hs = val / 100.0
                else:
                    hs = val
    except Exception:
        pass

    # Final safety
    if kpr is not None:
        try:
            kpr = float(kpr)
        except:
            kpr = None
    if hs is not None:
        try:
            hs = float(hs)
        except:
            hs = None

    return kpr, hs

def extract_first_match_id_from_matches_page(html: str):
    \"\"\"Return first match id on HLTV matches page: looks for '/matches/{id}/'\"\"\"
    m = re.search(r'/matches/(\\d+)', html)
    if m:
        return m.group(1)
    return None

def fetch_hltv_player_page(player_href: str):
    url = f\"https://www.hltv.org{player_href}\" if player_href.startswith(\"/player/\") else player_href
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    return r.text

def find_players_from_match_page(html: str):
    \"\"\"Attempt to find player names and profile links on a match page.\"\"\"
    soup = BeautifulSoup(html, 'html.parser')
    players = []
    # HLTV match pages often include player links with href like '/player/7592/device'
    for a in soup.select('a'):
        href = a.get('href','')
        if isinstance(href, str) and href.startswith('/player/') and a.get_text(strip=True):
            name = a.get_text(strip=True)
            players.append({'name': name, 'href': href})
    # Deduplicate by name preserving order
    seen = set()
    out = []
    for p in players:
        if p['name'] not in seen:
            seen.add(p['name'])
            out.append(p)
    return out

def fetch_match_page(match_id: str):
    url = f\"https://www.hltv.org/matches/{match_id}/\"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    return r.text

def fetch_todays_first_match_id():
    cached = cache_get('todays_first_match')
    if cached:
        return cached
    url = 'https://www.hltv.org/matches'
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    html = r.text
    match_id = extract_first_match_id_from_matches_page(html)
    if match_id:
        cache_set('todays_first_match', match_id)
    return match_id

def estimate_rounds_by_odds(match_html: str):
    \"\"\"Heuristic: inspect match page for odds or signs of a heavy favorite.
       Return rounds_per_map float (22-26).\"\"\"
    # Default average
    rounds = 23.5
    try:
        # Look for percent odds text like '60%' or '1.20' implied odds; fallback to simple heuristics
        # We'll search for team odds or betting indicators
        # If one team is clearly favored in text (e.g., contains 'favorite' or shows larger percent) choose 22/23
        if 'favorite' in match_html.lower() or 'favored' in match_html.lower():
            rounds = 22.0
        # look for large difference in win probability text like '60%' vs '40%'
        percents = re.findall(r'([0-9]{1,3})\\%', match_html)
        if percents and len(percents) >= 2:
            try:
                pnums = [int(p) for p in percents[:4]]
                diff = abs(pnums[0] - pnums[1])
                if diff >= 15:
                    rounds = 22.0
                elif diff <= 8:
                    rounds = 26.0
            except:
                pass
    except Exception:
        pass
    return rounds

# --- Main functions ---
def build_props_for_match(match_id: str):
    cache_key = f'match_props_{match_id}'
    cached = cache_get(cache_key)
    if cached:
        return cached

    match_html = fetch_match_page(match_id)
    if not match_html:
        return []

    rounds_per_map = estimate_rounds_by_odds(match_html)
    players = find_players_from_match_page(match_html)
    out = []
    # Limit to first 20 players to be safe
    for p in players[:20]:
        try:
            player_page = fetch_hltv_player_page(p['href'])
            if not player_page:
                continue
            kpr, hs = parse_player_stats_from_player_page(player_page)
            # expected kills per map based on kpr * rounds_per_map
            expected_kills_per_map = None
            if kpr is not None:
                expected_kills_per_map = round(kpr * rounds_per_map, 2)
            # Default kill_line/hs_line placeholders are expected metrics (frontend can replace with actual PrizePicks lines later)
            entry = {
                'player': p['name'],
                'player_href': p['href'],
                'kpr': kpr if kpr is not None else 0,
                'hs_percent': hs if hs is not None else 0,
                'expected_kills_per_map': expected_kills_per_map if expected_kills_per_map is not None else 0,
                'kill_line': expected_kills_per_map if expected_kills_per_map is not None else 0,
                'hs_line': round((hs if hs is not None else 0) * (expected_kills_per_map if expected_kills_per_map else 1), 2) if hs is not None and expected_kills_per_map else 0
            }
            out.append(entry)
        except Exception as e:
            continue

    cache_set(cache_key, out)
    return out

# --- Endpoints ---
@app.get('/match/props')
def get_default_match_props(match_id: Optional[str] = Query(None)):
    \"\"\"If match_id is provided, returns props for that match.
       Else uses today's first match on HLTV.\"\"\"
    try:
        if match_id:
            mid = match_id
        else:
            mid = fetch_todays_first_match_id()
            if not mid:
                raise HTTPException(status_code=502, detail='Could not find today\\'s first match on HLTV')
        props = build_props_for_match(mid)
        return {'match_id': mid, 'props': props}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/evaluate')
def evaluate_endpoint(body: EvaluateRequest):
    try:
        # Use provided kpr/hs_percent if present, else leave as-is
        kpr = body.kpr
        hs_pct = body.hs_percent
        # If not provided, try to infer from PrizePicks-like input (not required)
        map_count = body.map_count or 2
        # estimate rounds per map default 23.5; allow adjustment based on salary/other (not available here)
        rounds_per_map = 23.5
        expected_kills = None
        if kpr is not None:
            expected_kills = kpr * rounds_per_map * map_count
        elif body.kill_line is not None and float(body.kill_line) > 0:
            # If kpr missing but kill_line present, treat kill_line as expected kills and use that
            expected_kills = float(body.kill_line) * map_count
        else:
            expected_kills = 0.0

        # Value formula: upgraded blend
        hs_line_val = float(body.hs_line or 0)
        expected_kills_total = expected_kills
        value_score = round((hs_line_val * 0.65 + expected_kills_total * 0.35) - float(body.salary or 0), 2)

        verdict = '⚡ Smash over' if value_score >= 12.5 else '🤏 Mid play'

        return {
            'player': body.player,
            'value_score': value_score,
            'expected_kills': round(expected_kills_total, 2),
            'used_kpr': kpr,
            'used_hs': f\"{round((hs_pct or 0) * 100, 1)}%\" if hs_pct is not None else None,
            'verdict': verdict,
            'notes': 'Bolt upgraded formula (odds-adjusted rounds cached).'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def root():
    return {'message': 'Bolt backend — live'}
