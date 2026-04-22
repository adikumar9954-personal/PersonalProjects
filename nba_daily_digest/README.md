# NBA Daily Digest

A script that scrapes Reddit r/nba post-game threads each morning, summarizes standout performances using Gemini, and emails you a polished digest.

---

## What It Does

1. **Scrapes Reddit** — finds every post-game thread from the previous day's games on r/nba
2. **Summarizes with Gemini** — sends the thread content to Gemini 2.5 Flash, which picks out the best individual performances, the game of the night, and a hot take
3. **Emails you** — delivers a dark-mode HTML digest called "Today's Ballers" via Gmail SMTP

---

## Setup

**1. Install dependencies**
```bash
pip install requests google-generativeai
```

**2. Set environment variables**

Copy `.env.example` to `.env` and fill in your values:
```
GEMINI_API_KEY    — free at https://aistudio.google.com/app/apikey
EMAIL_SENDER      — your Gmail address
EMAIL_PASSWORD    — Gmail App Password (not your login password)
                    create one at https://myaccount.google.com/apppasswords
EMAIL_RECIPIENT   — where to send the digest (can be same as sender)
```

**3. Run it**
```bash
python nba_daily_digest.py
```

Run it each morning — it pulls the previous day's games by default (`days_back = 1`).

---

## Configuration

At the top of the script, the `CONFIG` block has two settings you might want to adjust:

| Setting | Default | What it does |
|---|---|---|
| `days_back` | `1` | How many days back to look for games. Use `1` in the morning, `0` if running late at night after games finish. |
| `max_comments_per_thread` | `50` | How many Reddit comments to send to Gemini per thread. Higher = more context, more tokens used. |
| `max_threads` | `15` | Max number of game threads to process in one run. |

---

