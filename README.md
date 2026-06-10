# pulse-pipeline

Scheduled Python pipeline that fetches AI news RSS feeds, clusters articles by story, generates summaries via NIM LLM, and writes to Supabase. Runs every 25 minutes via GitHub Actions.

## Setup

### 1. Add GitHub Actions Secrets

Go to **Settings → Secrets and variables → Actions** in your repo and add:

| Secret | Value |
|---|---|
| `SUPABASE_URL` | `https://[project-ref].supabase.co` |
| `SUPABASE_SERVICE_KEY` | Service role key from Supabase dashboard |
| `NIM_API_KEY` | NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com) |

### 2. Push to GitHub

```bash
git add .
git commit -m "init pulse-pipeline"
git push
```

The pipeline starts running automatically on the 25-minute cron.

### 3. Manual trigger

Go to **Actions → Pulse Pipeline → Run workflow** to trigger a run immediately without waiting for the next cron slot.

## Files

```
pipeline.py           # All 6 pipeline steps
sources.yaml          # RSS feed allowlist — edit to add/remove sources
requirements.txt      # Python dependencies
.github/workflows/
  pipeline.yml        # Main 25-min cron
  keepalive.yml       # Pings Supabase every 4 days (prevents free-tier pause)
```

## Adding sources

Edit `sources.yaml`. Each entry needs `name`, `feed_url`, and `tier` (1 or 2). No code changes needed.

## Monitoring

Failed runs trigger a GitHub email notification automatically. Run logs are in **Actions → Pulse Pipeline → [run]**.
