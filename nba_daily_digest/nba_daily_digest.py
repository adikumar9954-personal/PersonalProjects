"""
NBA Daily Performance Digest
=============================
Sources:  Reddit r/nba (post-game threads + game threads)
LLM:      Gemini 1.5 Flash (free tier — Google AI Studio)
Email:    Gmail SMTP

Setup:
  pip install requests google-generativeai

Environment variables (or fill in CONFIG below):
  GEMINI_API_KEY   - from https://aistudio.google.com/app/apikey
  EMAIL_SENDER     - your Gmail address
  EMAIL_PASSWORD   - Gmail App Password (NOT your login password)
                     Create one: myaccount.google.com/apppasswords
  EMAIL_RECIPIENT  - destination address (can be same as sender)
"""

import os
import re
import smtplib
import textwrap
import time
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import google.generativeai as genai

# ─────────────────────────── CONFIG ────────────────────────────────────────
CONFIG = {
    "gemini_api_key":  os.getenv("GEMINI_API_KEY",  ""),
    "email_sender":    os.getenv("EMAIL_SENDER",    ""),
    "email_password":  os.getenv("EMAIL_PASSWORD",  ""),
    "email_recipient": os.getenv("EMAIL_RECIPIENT", ""),
    # 1 = yesterday's games (best for a morning digest run).
    # Set to 0 if running late at night after games finish.
    "days_back": 1,
    # Max top-level comments to pull from each thread (Gemini context)
    "max_comments_per_thread": 50,
    # Max threads to process total
    "max_threads": 15,
}

# Reddit requires a descriptive User-Agent
REDDIT_HEADERS = {
    "User-Agent": "nba-digest-bot/1.0 (personal project; contact via reddit)"
}
# ───────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_target_date() -> date:
    return date.today() - timedelta(days=CONFIG["days_back"])


def reddit_get(url: str, params: dict = None) -> dict | None:
    """GET a Reddit JSON endpoint, respecting rate limits."""
    try:
        r = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
        if r.status_code == 429:
            print("  [Reddit] Rate limited — waiting 10s...")
            time.sleep(10)
            r = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [Reddit] Request failed for {url}: {e}")
        return None


def unix_to_date(ts: float) -> date:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


# ══════════════════════════════════════════════════════════════════════════════
#  REDDIT SCRAPING
# ══════════════════════════════════════════════════════════════════════════════

def search_rnba_threads(target: date) -> list[dict]:
    """
    Browse r/nba new feed to find post-game threads posted on `target`.
    Uses /new instead of search — much more reliable for recent threads.
    Returns raw Reddit post data dicts.
    """
    found = {}
    after = None   # pagination cursor

    # r/nba gets ~100-200 posts/day; scan up to 5 pages of 100 to find target date
    for page in range(5):
        params = {"limit": 100}
        if after:
            params["after"] = after

        data = reddit_get("https://www.reddit.com/r/nba/new.json", params=params)
        if not data:
            break

        posts = data.get("data", {}).get("children", [])
        if not posts:
            break

        oldest_date = None
        for post in posts:
            p = post["data"]
            post_date = unix_to_date(p["created_utc"])
            oldest_date = post_date  # last one will be oldest on this page

            title_lower = p.get("title", "").lower()
            is_pgt = "post game thread" in title_lower or "post-game thread" in title_lower
            if is_pgt and post_date == target:
                found[p["id"]] = p

        after = data.get("data", {}).get("after")

        # If the oldest post on this page is already before our target, stop paginating
        if oldest_date and oldest_date < target:
            break
        if not after:
            break

        time.sleep(1)

    print(f"[Reddit] Found {len(found)} post-game thread(s) for {target}")
    return list(found.values())


def fetch_thread_content(post: dict) -> dict | None:
    """
    Fetch the full post body + top comments for a Reddit thread.
    Returns {"title", "url", "source", "text"} or None.
    """
    permalink = "https://www.reddit.com" + post["permalink"] + ".json"
    data = reddit_get(permalink, params={"limit": CONFIG["max_comments_per_thread"], "depth": 1})
    if not data or len(data) < 2:
        return None

    title    = post.get("title", "Untitled")
    selftext = post.get("selftext", "").strip()

    # Collect + sort top-level comments by score
    comments_raw = data[1].get("data", {}).get("children", [])
    comments = []
    for c in comments_raw:
        body  = c.get("data", {}).get("body", "")
        score = c.get("data", {}).get("score", 0)
        if body and body not in ("[deleted]", "[removed]") and len(body) > 20:
            comments.append((score, body))

    comments.sort(key=lambda x: x[0], reverse=True)
    top_comments = [body for _, body in comments[: CONFIG["max_comments_per_thread"]]]

    parts = []
    if selftext:
        parts.append(f"[POST BODY]\n{selftext}")
    if top_comments:
        parts.append("[TOP COMMENTS]\n" + "\n---\n".join(top_comments))

    text = "\n\n".join(parts)
    if not text:
        return None

    return {
        "source": "Reddit r/nba",
        "title":  title,
        "url":    "https://www.reddit.com" + post["permalink"],
        "text":   text,
    }


def scrape_reddit(target: date) -> list[dict]:
    """Main Reddit scraping entry point."""
    print(f"\n[Reddit] Scraping r/nba for {target} ...")
    posts = search_rnba_threads(target)

    # Keep only post-game threads — filter out game threads and anything else
    posts = [
        p for p in posts
        if "post game thread" in p.get("title", "").lower()
        or "post-game thread" in p.get("title", "").lower()
    ]
    print(f"[Reddit] {len(posts)} post-game thread(s) after filtering")
    posts = posts[: CONFIG["max_threads"]]

    articles = []
    for post in posts:
        print(f"  Fetching: {post['title'][:80]}")
        content = fetch_thread_content(post)
        if content:
            articles.append(content)
            print(f"    ✓ {len(content['text'])} chars")
        time.sleep(0.75)  # stay within Reddit's ~1 req/sec guideline

    print(f"\n📰 Total Reddit threads collected: {len(articles)}\n")
    return articles


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def summarize_with_gemini(articles: list[dict], target: date) -> str:
    """Send thread content to Gemini; return a curated performance digest."""
    genai.configure(api_key=CONFIG["gemini_api_key"])
    model = genai.GenerativeModel("gemini-2.5-flash")

    if not articles:
        return (
            "No Reddit threads were found for this date.\n\n"
            "Tips:\n"
            "• Post-game threads usually appear within 30 min of the final buzzer.\n"
            "• Try days_back=0 late at night, or days_back=1 the next morning.\n"
            "• Confirm that your target date actually had NBA games scheduled."
        )

    # Build context — prioritise stat-rich lines so nothing gets cut
    context_parts = []
    for i, a in enumerate(articles, 1):
        text = a["text"]

        # Pull lines that look like stat lines and put them first
        lines = text.splitlines()
        stat_lines = [
            l for l in lines
            if re.search(r'\d+\s*(pts|reb|ast|stl|blk|points|rebounds|assists)', l, re.I)
            or re.search(r'\d+-\d+', l)   # shooting splits like 8-14
        ]
        other_lines = [l for l in lines if l not in stat_lines]

        # Stat lines first, then rest — cap at 6000 chars per thread
        reassembled = "\n".join(stat_lines) + "\n\n" + "\n".join(other_lines)
        snippet = reassembled[:6000]
        context_parts.append(f"=== Thread {i}: {a['title']} ===\n{snippet}")
    context = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""
        You are an NBA analyst writing a daily digest email called "🏀 Today's Ballers".

        Date: {target.strftime("%A, %B %d, %Y")}

        Below are excerpts from Reddit r/nba post-game threads and game threads
        for games played on {target.strftime("%B %d, %Y")}.
        Content includes official recap text AND fan comments highlighting key moments.

        YOUR TASKS:

        1. **Standout Performances** — Identify the 3–7 most impressive individual
           player performances. For EACH player write:
           • **[Player Name] — [Team]**
           • Stat line: points / rebounds / assists / steals / blocks / etc.
           • 2–3 sentences on WHY it was impressive: opponent quality, clutch moments,
             efficiency, context (comeback, rivalry, career milestone, etc.)

        2. **Game of the Night** — Pick the most exciting or significant game.
           One short paragraph explaining what made it special.

        3. **Hot Take / Fun Fact** — End with 1–2 witty sentences: a spicy observation,
           a funny Reddit reaction that stood out, or a surprising stat.

        RULES:
        - Only use stats and facts present in the source text. Do NOT invent numbers.
        - Always include the stat line even if incomplete (e.g. '28 pts, X reb, X ast' is fine).
        - Look for shooting splits (e.g. 10-18 FG, 4-7 3P) in the thread body — include them.
        - If exact stats are missing, use what's available and note it's partial.
        - Be energetic and fun — this is a newsletter, not a box score dump.
        - Use markdown formatting (##, ###, **bold**) so the email renders nicely.

        SOURCE THREADS:
        {context}
    """)

    print("[Gemini] Generating digest...")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini summarization failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def build_html_email(digest_text: str, articles: list[dict], target: date) -> str:
    """Wrap the Gemini digest in a polished dark-mode HTML email."""
    date_label = target.strftime("%A, %B %d, %Y")

    html_body = digest_text
    html_body = re.sub(r"^###\s*(.+)$",  r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^##\s*(.+)$",   r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^#\s*(.+)$",    r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", html_body)
    html_body = re.sub(r"^[-•]\s+(.+)$", r"<li>\1</li>", html_body, flags=re.MULTILINE)
    html_body = html_body.replace("\n", "<br>")

    sources_html = "".join(
        f'<li><a href="{a["url"]}">{a["title"][:90]}</a></li>'
        for a in articles
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin:0; padding:20px;
      background:#0a0a0a;
      font-family:'Georgia',serif;
      color:#e8e8e8;
    }}
    .container {{
      max-width:680px; margin:0 auto;
      background:#111; border-radius:16px;
      overflow:hidden; box-shadow:0 8px 40px rgba(0,0,0,.6);
    }}
    .header {{
      background:linear-gradient(135deg,#c8102e 0%,#1d428a 100%);
      padding:40px 36px 32px;
    }}
    .header h1 {{ margin:0 0 6px; font-size:30px; color:#fff; letter-spacing:-.5px; }}
    .header .sub {{ color:rgba(255,255,255,.7); font-size:14px; margin:0; }}
    .body {{ padding:36px; line-height:1.8; font-size:15px; color:#ccc; }}
    .body h2 {{
      color:#f5a623; font-size:19px;
      border-bottom:1px solid #2a2a2a;
      padding-bottom:8px; margin-top:32px;
    }}
    .body h3 {{ color:#7fb3f5; font-size:16px; margin-top:24px; margin-bottom:4px; }}
    .body strong {{ color:#fff; }}
    .body em {{ color:#aaa; font-style:italic; }}
    .body li {{ margin-bottom:4px; }}
    hr.div {{ border:none; border-top:1px solid #1e1e1e; margin:0; }}
    .sources {{ background:#0c0c0c; padding:20px 36px; }}
    .src-label {{
      font-size:10px; text-transform:uppercase;
      letter-spacing:2px; color:#444; margin:0 0 10px;
    }}
    .sources ul {{ margin:0; padding:0 0 0 14px; }}
    .sources li {{ font-size:12px; margin-bottom:5px; }}
    .sources a {{ color:#4a4a6a; text-decoration:none; }}
    .footer {{ text-align:center; padding:18px; font-size:11px; color:#333; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🏀 Today's Ballers</h1>
      <p class="sub">NBA Standout Performances &mdash; {date_label}</p>
    </div>
    <div class="body">{html_body}</div>
    <hr class="div">
    <div class="sources">
      <p class="src-label">Reddit r/nba Sources</p>
      <ul>{sources_html}</ul>
    </div>
    <div class="footer">Generated automatically from r/nba &middot; NBA Daily Digest</div>
  </div>
</body>
</html>"""


def send_email(subject: str, html_content: str, plain_text: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = CONFIG["email_sender"]
    msg["To"]      = CONFIG["email_recipient"]
    msg.attach(MIMEText(plain_text,   "plain"))
    msg.attach(MIMEText(html_content, "html"))

    print(f"[Email] Connecting to Gmail SMTP...")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(CONFIG["email_sender"], CONFIG["email_password"])
            server.sendmail(CONFIG["email_sender"], CONFIG["email_recipient"], msg.as_string())
        print(f"[Email] ✅ Sent to {CONFIG['email_recipient']}")
    except smtplib.SMTPAuthenticationError:
        print(
            "[Email] ✗ Authentication failed.\n"
            "  Use a Gmail App Password, not your regular login password.\n"
            "  Create one at: https://myaccount.google.com/apppasswords"
        )
        raise
    except Exception as e:
        print(f"[Email] ✗ Failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    target = get_target_date()
    print(f"\n🏀 NBA Daily Digest — {target.strftime('%B %d, %Y')}")
    print("=" * 52)

    # 1. Scrape Reddit
    articles = scrape_reddit(target)

    # 2. Summarize with Gemini
    digest = summarize_with_gemini(articles, target)
    print("\n── DIGEST PREVIEW ──────────────────────────────────")
    print(digest[:800] + ("...\n[truncated]" if len(digest) > 800 else ""))
    print("────────────────────────────────────────────────────\n")

    # 3. Build & send email
    subject = f"🏀 NBA Ballers Digest — {target.strftime('%b %d, %Y')}"
    html    = build_html_email(digest, articles, target)
    send_email(subject, html, digest)


if __name__ == "__main__":
    main()