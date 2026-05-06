# Day 8 Setup: Email + Notion Delivery

Goal tonight: briefings land in your Gmail inbox and a Notion database every morning, with zero manual steps.

## What got built

1. `digest.py` — parses `briefings/YYYY-MM-DD/*.md` into structured fields (summary, key_signals, watch, sources).
2. Two new endpoints on `pipeline_server.py`:
   - `GET /deliver` — JSON with all 5 briefings, used by Notion workflow.
   - `GET /deliver-html` — pre-rendered email body, used by Gmail workflow.
3. `n8n/workflows/05_email_digest.json` — cron 8am, hits `/deliver-html`, sends one HTML email.
4. `n8n/workflows/06_notion_briefings.json` — cron 8:05am, hits `/deliver`, creates 5 Notion pages.

## Step 1: Restart pipeline_server.py

The server has new endpoints. Kill the old one and restart so the new routes are live.

```bash
# in the project dir, with venv active
pkill -f pipeline_server.py
python pipeline_server.py
```

Sanity check the new endpoints. Use a recent date you have briefings for:

```bash
curl 'http://localhost:8000/deliver?date=2026-05-01' | jq '.count, .briefings[0].ticker'
curl 'http://localhost:8000/deliver-html?date=2026-05-01' | jq '.subject'
```

Expected output: `5`, `"AAPL"`, then a subject line.

## Step 2: Get a Gmail App Password (5 min)

Gmail SMTP needs an App Password, not your normal password. This requires 2FA on the account.

1. Go to https://myaccount.google.com/security
2. Confirm 2-Step Verification is on. If not, turn it on first.
3. Go to https://myaccount.google.com/apppasswords
4. App name: `n8n AMIA`
5. Click Create. Copy the 16-character password.
6. Save it somewhere safe. You will not see it again.

## Step 3: Create the Gmail SMTP credential in n8n

1. Open n8n at http://localhost:5678
2. Top right, click your initials → Credentials
3. Click "Add credential" → search "SMTP" → pick "SMTP"
4. Fill in:
   - User: `wong.kittituch@gmail.com`
   - Password: the App Password from Step 2 (paste with no spaces)
   - Host: `smtp.gmail.com`
   - Port: `465`
   - SSL/TLS: on
5. Name it `Gmail SMTP`. Save.
6. Click "Test" — n8n will try to connect. Should say success.

## Step 4: Set up Notion (10 min)

### 4a. Create the integration

1. Go to https://www.notion.so/my-integrations
2. Click "New integration"
3. Name: `AMIA`
4. Associated workspace: pick your personal workspace
5. Type: Internal
6. Capabilities: Read, Update, Insert content (default is fine)
7. Submit. Copy the "Internal Integration Secret" — starts with `ntn_`.

### 4b. Create the AMIA Briefings database

1. In Notion, create a new page called `AMIA Briefings`.
2. Inside that page, type `/database` and pick "Database - Full page".
3. Add these properties (the names matter, the workflow uses them):
   - `Name` — Title (default, already there)
   - `Ticker` — Select. Options: AAPL, TSLA, NVDA, MSFT, AMZN
   - `Date` — Date
   - `Preview` — Text
4. Top right of the database page → click `...` → Connections → search `AMIA` → Add. This grants the integration access to this DB.

### 4c. Grab the database ID

Open the database as a full page. The URL looks like:

```
https://www.notion.so/<workspace>/<32-char-database-id>?v=<view-id>
```

Copy the 32-character string (no dashes) between the workspace name and the `?v=`. That's your database ID.

### 4d. Create the Notion credential in n8n

1. n8n → Credentials → Add credential → search "Notion API"
2. Internal Integration Secret: paste the `ntn_...` token
3. Name: `Notion AMIA`. Save.
4. Test it. Should say success.

## Step 5: Import the two workflows

1. n8n → top left → Workflows → Add workflow → Import from file
2. Pick `n8n/workflows/05_email_digest.json`. Save.
3. Open the "Send Gmail" node → re-pick the `Gmail SMTP` credential from the dropdown (the imported ID is a placeholder).
4. Save the workflow. Toggle "Active" on the top right.
5. Repeat for `06_notion_briefings.json`:
   - Open "Create Notion page" node
   - Re-pick the `Notion AMIA` credential
   - Open the "Database" dropdown, pick `AMIA Briefings`. n8n will refresh property keys.
   - Confirm the Title and Property mappings still point at `ticker`, `date`, `preview` from the previous nodes.
   - Save and activate.

## Step 6: End-to-end test (manual fire)

Don't wait until 8am tomorrow. Force a run now.

1. Open Workflow 5 → top right, click "Execute Workflow" (manual trigger).
2. Watch the nodes turn green. Last node should show "Success" with a single output item.
3. Check your Gmail inbox. Subject should be `AMIA Daily Briefing - 2026-05-01 (5 tickers)`.
4. Open Workflow 6 → "Execute Workflow".
5. Notion DB should have 5 new pages, one per ticker.

## Common things that break

1. **`ECONNREFUSED 127.0.0.1:8000` from n8n**
   You're using `localhost` somewhere. Inside Docker, `localhost` is the container, not your Mac. The workflows already use `host.docker.internal:8000`. If you see this error, double-check the HTTP Request node URL.

2. **Notion property "Ticker" not found**
   Property name mismatch. Notion is case-sensitive. Confirm the column header is exactly `Ticker`, not `ticker` or `Tickers`.

3. **Gmail says "Username and Password not accepted"**
   App Password was pasted with spaces, or 2FA isn't on. Generate a new App Password and paste it as one block of 16 chars.

4. **Notion API returns "object_not_found"**
   You forgot to grant the integration access to the database. Open the DB page → `...` → Connections → add AMIA.

5. **Email arrives but renders as raw HTML tags**
   The Send Gmail node is in plain-text mode. Open it, set "Email Format" to HTML.

## Day 8 EOD checklist

- [ ] `/deliver` and `/deliver-html` return real data
- [ ] Gmail App Password generated, SMTP credential tests green in n8n
- [ ] Notion integration created, AMIA Briefings DB shared with it
- [ ] Workflow 5 imported, Gmail credential re-picked, manual run sends an email
- [ ] Workflow 6 imported, Notion credential re-picked, manual run creates 5 DB pages
- [ ] Both workflows toggled Active so they fire at 8am Sydney tomorrow

## What's next (Day 9 preview)

Self-hosted Langfuse for tracing every LangChain / LangGraph / CrewAI call. We'll spin it up via Docker Compose, wire callback handlers into the agent code, and find at least one weak prompt or wasteful retrieval to fix on Day 11.
