import os
import asyncio
import requests
import anthropic
import pymongo
import discord
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
import functions

if os.path.exists("env.py"):
    import env

anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openrouter_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=90.0
)

ROAST_MODEL = "mistralai/mistral-small-3.2-24b-instruct"

# ─── ROSTER MAP ───────────────────────────────────────────────────────────────
ROSTER_ID_MAP = {
    1:  "Jake Boy",     # Gravylegs69
    2:  "Joe T",        # joeschmoe9
    3:  "Rob H",        # RHawdogreality
    4:  "Caleb G",      # ctgaloozis
    5:  "Joe L",        # jplibs
    6:  "Jim D",        # jimdeng
    7:  "Rohan P",      # rohandynasty
    8:  "Gaspar A",     # Gaspy9000
    9:  "Jake Barann",  # jbarann
    10: "Nick A",       # nlain
    11: "Ricky K",      # LouSassole43 — joined for first Sleeper season, no ESPN history
    12: "Lowell F",     # FondlerMagnusson
}

_NAME_TO_ROSTER_ID = {v.lower(): k for k, v in ROSTER_ID_MAP.items()}

# ─── ESPN CONFIG ──────────────────────────────────────────────────────────────
ESPN_LEAGUE_ID  = os.environ.get("ESPN_LEAGUE_ID", "")
ESPN_S2_COOKIE  = os.environ.get("ESPN_S2", "")
ESPN_SWID_COOKIE = os.environ.get("ESPN_SWID", "")
ESPN_SEASONS    = [2019, 2021, 2022, 2023]

ESPN_OWNER_TO_ROSTER_ID = {
    "Slender Foot":    1,
    "Joseph Tribou":   2,
    "Rob Hawthorne":   3,
    "Caleb Galoozis":  4,
    "Joe Liberty":     5,
    "Jim Deng":        6,
    "Rohan Pidaparti": 7,
    "gaspar avila":    8,
    "Jake Barann":     9,
    "Nicholas Allain": 10,
    "Lowell Fluke":    12,
}

# ─── PERSONAS ─────────────────────────────────────────────────────────────────
ROAST_PERSONA = """You are roasting managers in a private fantasy football league. This is adult roast comedy — no network, no HR, no filter. These friends built this bot specifically to destroy each other.

Tone and energy: Shane Gillis roasting his boys in a basement. Jeff Ross at the Friar's Club. Tony Hinchcliffe with no producer notes. Mean, specific, and actually funny — not "edgy for an AI" funny.

FANTASY FOOTBALL CONTEXT (use this to interpret the stats):
- A good weekly score is 120+ pts. Under 100 is weak. Under 85 is humiliating.
- Season PF above 1550 is strong. Below 1400 is pathetic.
- "Left X pts on bench" = they started worse players while better ones sat. This is stupidity, not bad luck.
- When a starter scores under 5 pts, that's a catastrophically bad lineup call.
- When a bench player scores 20+ while sitting, that's unforgivable.
- 9-5 is a solid record. 5-9 or worse means they're genuinely bad.
- 0 championships in multiple seasons = chronic loser. Use this.

Format: 5-7 short lines. Each line its own paragraph with a blank line between. Punchlines, not paragraphs.

Rules:
- Profanity not just allowed — expected. Get filthy.
- Every line must use a specific stat, player name, record, or history fact from the data. No line should be detachable from this specific person.
- No metaphors that have nothing to do with fantasy football. Keep the insults grounded in what actually happened.
- No warmup. Start swinging immediately.
- Do not compliment them at the end. Do not soften the landing. End mean."""

RECAP_PERSONA = """You are the weekly recap writer for a private fantasy football league. Unhinged sports radio host who hates everyone.

FANTASY FOOTBALL CONTEXT:
- Good score = 120+ pts. Bad score = under 90. Embarrassing = under 80.
- Bench waste = points left on bench from players they didn't start. Stupidity, not luck.
- Close losses are funnier than blowouts. Mock the margin.

- Call out losers by name and make it hurt with specific numbers
- Celebrate winners just enough to make losers feel worse
- Profanity expected. Deranged Bill Simmons energy.
- Under 300 words. Every sentence makes someone uncomfortable."""

POWER_RANKINGS_PERSONA = """Weekly power rankings for a private fantasy football league. Brutal analyst who hates bad managers.

- 1-2 sentences per team. Make every word count.
- Top teams: grudging respect. Bottom teams: destroyed with specifics.
- No hedging. No filler.
- Number 1 through total teams.
- Last place should feel genuinely terrible."""

LEAGUE_ROAST_PERSONA = """Roasting every team in a private fantasy football league at once.

FANTASY CONTEXT: Good score = 120+. Under 90 = embarrassing. 0 championships = chronic loser.

FORMAT per team:
**[rank]. [Name]**
[2-3 savage lines using their specific stats and history]

- Use the ESPN history where shown — years of failure is gold
- Profanity expected
- Move fast between teams, no warmup
- Worst teams get it worst. Nobody escapes."""

# ─── INFRASTRUCTURE ───────────────────────────────────────────────────────────
_username_cache  = {}
_espn_history_cache = {}
_sleeper_players = {}   # player_id → full name, loaded at startup
_executor = ThreadPoolExecutor(max_workers=10)


def get_mongo():
    return pymongo.MongoClient(os.environ.get("MONGO_URI"))[os.environ.get("MONGO_DBNAME")]


async def fetch_json(url, cookies=None, headers=None):
    loop = asyncio.get_event_loop()
    def _fetch():
        return requests.get(url, cookies=cookies, headers=headers).json()
    return await loop.run_in_executor(_executor, _fetch)


async def get_sleeper_data_parallel(league_id):
    rosters, users, nfl_state = await asyncio.gather(
        fetch_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters"),
        fetch_json(f"https://api.sleeper.app/v1/league/{league_id}/users"),
        fetch_json("https://api.sleeper.app/v1/state/nfl")
    )
    return rosters, users, nfl_state.get("week", 1), nfl_state.get("season_type", "regular")


async def get_matchups_parallel(league_id, weeks):
    if not weeks:
        return {}
    results = await asyncio.gather(*[
        fetch_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{w}")
        for w in weeks
    ])
    return {w: r for w, r in zip(weeks, results)}


async def get_last_scored_week(league_id, current_week, season_type):
    if season_type == "regular" and current_week > 1:
        return current_week - 1
    for w in range(18, 0, -1):
        try:
            matchups = await fetch_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{w}")
            if matchups and any(m.get("points", 0) > 0 for m in matchups):
                return w
        except Exception:
            continue
    return 17


def get_player_name(player_id):
    """Look up player name from the in-memory Sleeper player cache."""
    return _sleeper_players.get(str(player_id), f"Player#{player_id}")


def get_display_name(roster_id):
    return ROSTER_ID_MAP.get(roster_id, f"Roster {roster_id}")


def get_roster_id_by_name(name):
    return _NAME_TO_ROSTER_ID.get(name.lower())


# ─── SLEEPER PLAYER CACHE ─────────────────────────────────────────────────────
async def load_sleeper_players():
    """Load all NFL player names from Sleeper API into memory at startup."""
    global _sleeper_players
    try:
        data = await fetch_json("https://api.sleeper.app/v1/players/nfl")
        for pid, p in data.items():
            full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full:
                _sleeper_players[pid] = full
        print(f"Loaded {len(_sleeper_players)} player names from Sleeper")
    except Exception as e:
        print(f"Failed to load Sleeper players: {e}")


# ─── ESPN HISTORY ─────────────────────────────────────────────────────────────
def _espn_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://fantasy.espn.com/",
        "x-fantasy-source": "kona",
    }

def _espn_cookies():
    return {"espn_s2": ESPN_S2_COOKIE, "SWID": ESPN_SWID_COOKIE}


async def fetch_espn_season(year):
    if not ESPN_LEAGUE_ID or not ESPN_S2_COOKIE:
        return None
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
        f"/seasons/{year}/segments/0/leagues/{ESPN_LEAGUE_ID}"
        f"?view=mStandings&view=mTeam&view=mSettings"
    )
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: requests.get(url, cookies=_espn_cookies(), headers=_espn_headers()).json()
        )
    except Exception as e:
        print(f"ESPN fetch failed for {year}: {e}")
        return None


def parse_espn_season(data, year):
    if not data or "teams" not in data:
        return {}
    members = {m["id"]: f"{m.get('firstName','')} {m.get('lastName','')}".strip()
               for m in data.get("members", [])}
    results = {}
    for team in data.get("teams", []):
        rec = team.get("record", {}).get("overall", {})
        wins  = rec.get("wins", 0)
        losses = rec.get("losses", 0)
        pf    = round(rec.get("pointsFor", 0), 1)
        pa    = round(rec.get("pointsAgainst", 0), 1)
        final_rank = (team.get("rankCalculatedFinal") or
                      team.get("rankFinal") or
                      team.get("playoffSeed") or 0)
        owner_name = members.get(team.get("primaryOwner", ""), team.get("name", "Unknown"))
        results[owner_name] = {
            "season": year, "record": f"{wins}-{losses}",
            "pf": pf, "pa": pa,
            "final_rank": final_rank,
            "champion": final_rank == 1,
        }
    return results


async def load_espn_history():
    global _espn_history_cache
    if _espn_history_cache or not ESPN_LEAGUE_ID:
        return
    loaded = 0
    for year in ESPN_SEASONS:
        data = await fetch_espn_season(year)
        if not data:
            continue
        for owner_name, stats in parse_espn_season(data, year).items():
            roster_id = ESPN_OWNER_TO_ROSTER_ID.get(owner_name)
            if not roster_id:
                continue
            dn = get_display_name(roster_id)
            _espn_history_cache.setdefault(dn, {})[year] = stats
            loaded += 1
    print(f"ESPN history loaded: {loaded} team-seasons")


def get_historical_summary(display_name):
    history = _espn_history_cache.get(display_name, {})
    if not history:
        return ""
    lines = []
    championships = []
    bottom_half_count = 0
    total_seasons = len(history)
    total_teams_typical = 12

    for year in sorted(history.keys()):
        s = history[year]
        rank_str = f"#{s['final_rank']}" if s["final_rank"] else "?"
        champ = " 🏆 CHAMPION" if s["champion"] else ""
        lines.append(f"{year}: {s['record']}, {s['pf']} PF, finished {rank_str}{champ}")
        if s["champion"]:
            championships.append(str(year))
        if s["final_rank"] and s["final_rank"] > 6:
            bottom_half_count += 1

    summary = "ESPN career history:\n" + "\n".join(lines)
    if championships:
        summary += f"\nChampionships: {', '.join(championships)}"
    else:
        summary += f"\nChampionships: ZERO in {total_seasons} seasons"
    if bottom_half_count >= 2:
        summary += f"\nFinished bottom half {bottom_half_count}/{total_seasons} seasons"
    return summary


# ─── STARTUP ──────────────────────────────────────────────────────────────────
async def preload_username_cache(bot):
    try:
        MONGO = get_mongo()
        for server in MONGO.servers.find({"league": {"$exists": True}}):
            sid = server.get("server")
            if sid:
                _username_cache[sid] = list(ROSTER_ID_MAP.values())
                print(f"Loaded {len(_username_cache[sid])} names for server {sid}")
    except Exception as e:
        print(f"Failed to preload username cache: {e}")

    # Load these in parallel — player names and ESPN history
    await asyncio.gather(load_sleeper_players(), load_espn_history())


async def get_league_usernames(ctx: discord.AutocompleteContext):
    server_id = str(ctx.interaction.guild_id)
    return _username_cache.get(server_id, list(ROSTER_ID_MAP.values()))


# ─── ROAST DATA BUILDER ───────────────────────────────────────────────────────
def build_roster_roast_block(roster_id, display_name, roster, sorted_rosters, total_teams, recent_matchup):
    settings = roster.get("settings", {})
    wins  = settings.get("wins", 0)
    losses = settings.get("losses", 0)
    pf = float(f"{settings.get('fpts', 0)}.{settings.get('fpts_decimal', 0):02d}")
    pa = float(f"{settings.get('fpts_against', 0)}.{settings.get('fpts_against_decimal', 0):02d}")
    rank = next((i + 1 for i, r in enumerate(sorted_rosters) if r.get("roster_id") == roster_id), "?")

    record_context = "above .500" if wins > losses else ("below .500 — losing record" if wins < losses else ".500 — perfectly mediocre")

    lines = [
        f"Manager: {display_name}",
        f"Record: {wins}-{losses} ({record_context}, #{rank} of {total_teams})",
        f"Season points: {pf} scored, {pa} allowed",
    ]

    if recent_matchup:
        my_pts   = recent_matchup.get("points", 0)
        starters = recent_matchup.get("starters", [])
        s_pts    = recent_matchup.get("starters_points", [])
        all_pts  = recent_matchup.get("players_points", {})

        # Filter out 0-point starters (bye/IR) for cleaner analysis
        active_starters = [(pid, pts) for pid, pts in zip(starters, s_pts) if pts > 0]
        bench_entries   = [(pid, all_pts[pid]) for pid in all_pts if pid not in starters and all_pts[pid] > 0]
        bench_entries.sort(key=lambda x: x[1], reverse=True)
        bench_total = round(sum(v for _, v in bench_entries), 1)

        score_context = "decent" if my_pts >= 120 else ("weak" if my_pts >= 90 else "embarrassing")
        lines.append(f"Last week: {my_pts:.1f} pts ({score_context}), left {bench_total:.1f} pts sitting on bench")

        if active_starters:
            active_starters.sort(key=lambda x: x[1])
            worst_pid, worst_pts = active_starters[0]
            best_pid,  best_pts  = active_starters[-1]
            worst_name = get_player_name(worst_pid)
            best_name  = get_player_name(best_pid)
            if worst_pts < 8:
                lines.append(f"Started {worst_name} for {worst_pts:.1f} pts — that's a starting lineup decision, not bad luck")
            if best_pts > 0:
                lines.append(f"Best starter was {best_name} with {best_pts:.1f} pts")

        if bench_entries:
            top_bench_name = get_player_name(bench_entries[0][0])
            top_bench_pts  = bench_entries[0][1]
            if top_bench_pts > 15:
                lines.append(f"Left {top_bench_name} ({top_bench_pts:.1f} pts) rotting on the bench")
            if len(bench_entries) > 1 and bench_entries[1][1] > 12:
                lines.append(f"Also left {get_player_name(bench_entries[1][0])} ({bench_entries[1][1]:.1f} pts) sitting")

    history = get_historical_summary(display_name)
    if history:
        lines.append(history)

    return "\n".join(lines)


async def generate_single_roast(roast_block):
    prompt = (
        "Roast this fantasy football manager. Every insult must connect to a specific "
        "number, player name, or history line given below. No generic clichés.\n\n"
        + roast_block
    )
    response = await openrouter_client.chat.completions.create(
        model=ROAST_MODEL,
        messages=[
            {"role": "system", "content": ROAST_PERSONA},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=350,
        temperature=0.95
    )
    return response.choices[0].message.content.strip()


# ─── COMMANDS ─────────────────────────────────────────────────────────────────
async def roast_manager(ctx, bot, names: list):
    existing_league = functions.get_existing_league(ctx)
    if not existing_league or "league" not in existing_league:
        return "No league connected. Run /add-league first."

    league_id = existing_league["league"]
    rosters, users, current_week, season_type = await get_sleeper_data_parallel(league_id)
    last_week = await get_last_scored_week(league_id, current_week, season_type)
    last_week_matchups = (await get_matchups_parallel(league_id, [last_week])).get(last_week, [])

    sorted_rosters = sorted(rosters, key=lambda r: (-r["settings"].get("wins", 0), -r["settings"].get("fpts", 0)))
    total_teams = len(rosters)
    roast_blocks = []
    valid_names  = []

    for name in names:
        roster_id = get_roster_id_by_name(name)
        if roster_id is None:
            continue
        roster = next((r for r in rosters if r.get("roster_id") == roster_id), None)
        if not roster:
            continue
        recent_matchup = next((m for m in last_week_matchups if m.get("roster_id") == roster_id), None)
        block = build_roster_roast_block(roster_id, name, roster, sorted_rosters, total_teams, recent_matchup)
        roast_blocks.append(block)
        valid_names.append(name)

    if not roast_blocks:
        return "Couldn't find any of those managers."

    try:
        roasts = await asyncio.gather(*[generate_single_roast(b) for b in roast_blocks])
    except Exception as e:
        return f"Roast generator error: {str(e)}"

    sections = [f"🔥 **{name}**\n\n{roast}" for name, roast in zip(valid_names, roasts)]
    return "\n\n---\n\n".join(sections)


async def roast_league(ctx, bot):
    existing_league = functions.get_existing_league(ctx)
    if not existing_league or "league" not in existing_league:
        return "No league connected. Run /add-league first."

    league_id = existing_league["league"]
    rosters, users, current_week, season_type = await get_sleeper_data_parallel(league_id)
    last_week = await get_last_scored_week(league_id, current_week, season_type)
    last_week_matchups = (await get_matchups_parallel(league_id, [last_week])).get(last_week, [])

    sorted_rosters = sorted(rosters, key=lambda r: (-r["settings"].get("wins", 0), -r["settings"].get("fpts", 0)))
    team_lines = []

    for i, roster in enumerate(sorted_rosters):
        rid = roster.get("roster_id")
        dn  = get_display_name(rid)
        s   = roster.get("settings", {})
        wins   = s.get("wins", 0)
        losses = s.get("losses", 0)
        pf = float(f"{s.get('fpts',0)}.{s.get('fpts_decimal',0):02d}")

        recent = next((m for m in last_week_matchups if m.get("roster_id") == rid), None)
        last_score  = recent.get("points", 0) if recent else 0
        bench_waste = 0
        top_sit     = ""

        if recent:
            starters = recent.get("starters", [])
            all_pts  = recent.get("players_points", {})
            bench    = [(pid, v) for pid, v in all_pts.items() if pid not in starters and v > 0]
            bench.sort(key=lambda x: x[1], reverse=True)
            bench_waste = round(sum(v for _, v in bench), 1)
            if bench:
                top_sit = f"{get_player_name(bench[0][0])} ({bench[0][1]:.1f} pts sitting)"

        line = f"{i+1}. {dn} — {wins}-{losses}, {pf} PF, last week: {last_score:.1f}, bench waste: {bench_waste}"
        if top_sit:
            line += f", left {top_sit}"

        hist = _espn_history_cache.get(dn, {})
        if hist:
            champs = [str(y) for y, s2 in hist.items() if s2.get("champion")]
            short  = " | ".join(f"{y}: {s2['record']}" + (" 🏆" if s2.get("champion") else "") for y, s2 in sorted(hist.items()))
            line  += f"\n   History: {short}"
            if not champs:
                line += f" — 0 championships in {len(hist)} ESPN seasons"

        team_lines.append(line)

    prompt = "Roast every team. Use their specific stats and history. Short and savage:\n\n" + "\n".join(team_lines)

    try:
        response = await openrouter_client.chat.completions.create(
            model=ROAST_MODEL,
            messages=[
                {"role": "system", "content": LEAGUE_ROAST_PERSONA},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"League roast error: {str(e)}"


async def weekly_recap(ctx, bot, week=None):
    existing_league = functions.get_existing_league(ctx)
    if not existing_league or "league" not in existing_league:
        return "No league connected. Run /add-league first."

    league_id = existing_league["league"]
    rosters, users, current_week, season_type = await get_sleeper_data_parallel(league_id)
    target_week = int(week) if week is not None else await get_last_scored_week(league_id, current_week, season_type)
    matchups = await fetch_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{target_week}")
    roster_id_to_name = {r["roster_id"]: get_display_name(r["roster_id"]) for r in rosters}

    grouped = {}
    for m in matchups:
        grouped.setdefault(m["matchup_id"], []).append(m)

    result_lines = []
    all_scores   = []

    for mid, pair in grouped.items():
        if len(pair) != 2:
            continue
        a, b = pair
        a_name = roster_id_to_name.get(a["roster_id"], "Unknown")
        b_name = roster_id_to_name.get(b["roster_id"], "Unknown")
        a_pts  = a.get("points", 0)
        b_pts  = b.get("points", 0)

        w_name, w_pts, l_name, l_pts = (a_name, a_pts, b_name, b_pts) if a_pts > b_pts else (b_name, b_pts, a_name, a_pts)

        a_bench = round(sum(v for k, v in a.get("players_points", {}).items() if k not in a.get("starters", []) and v > 0), 1)
        b_bench = round(sum(v for k, v in b.get("players_points", {}).items() if k not in b.get("starters", []) and v > 0), 1)

        result_lines.append(
            f"{w_name} def. {l_name} {w_pts:.1f}-{l_pts:.1f} | bench waste: {a_name} {a_bench}, {b_name} {b_bench}"
        )
        all_scores += [(a_name, a_pts), (b_name, b_pts)]

    all_scores.sort(key=lambda x: x[1], reverse=True)
    high = all_scores[0]  if all_scores else ("?", 0)
    low  = all_scores[-1] if all_scores else ("?", 0)

    recap_prompt = f"Week {target_week}:\n{chr(10).join(result_lines)}\nHigh: {high[0]} {high[1]:.1f} | Low: {low[0]} {low[1]:.1f}"

    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=RECAP_PERSONA,
            messages=[{"role": "user", "content": recap_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Recap error: {str(e)}"


async def power_rankings(ctx, bot):
    existing_league = functions.get_existing_league(ctx)
    if not existing_league or "league" not in existing_league:
        return "No league connected. Run /add-league first."

    league_id = existing_league["league"]
    rosters, users, current_week, season_type = await get_sleeper_data_parallel(league_id)

    weeks = list(range(max(1, current_week - 3), current_week))
    matchups_by_week = await get_matchups_parallel(league_id, weeks)

    recent_pf = {r["roster_id"]: 0.0 for r in rosters}
    for w_matchups in matchups_by_week.values():
        for m in w_matchups:
            rid = m.get("roster_id")
            if rid in recent_pf:
                recent_pf[rid] += m.get("points", 0)

    ranked = []
    for r in rosters:
        rid = r.get("roster_id")
        s   = r.get("settings", {})
        wins   = s.get("wins", 0)
        losses = s.get("losses", 0)
        pf     = float(f"{s.get('fpts',0)}.{s.get('fpts_decimal',0):02d}")
        recent = round(recent_pf.get(rid, 0), 2)
        ranked.append({
            "name":      get_display_name(rid),
            "record":    f"{wins}-{losses}",
            "pf":        pf,
            "recent_pf": recent,
            "composite": (wins * 100) + pf + (recent * 0.5)
        })

    ranked.sort(key=lambda x: x["composite"], reverse=True)
    rankings_str = "\n".join([
        f"{i+1}. {t['name']} — {t['record']}, {t['pf']} PF, {t['recent_pf']} last 3 weeks"
        for i, t in enumerate(ranked)
    ])

    try:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=POWER_RANKINGS_PERSONA,
            messages=[{"role": "user", "content": rankings_str}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Power rankings error: {str(e)}"