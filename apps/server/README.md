# Reference HTTP server

A minimal Flask wrapper around the `agentstan` library: generate specs from
natural language, run simulations with polling, intervene mid-run, batch,
analyze, interpret.

**This is a demo, not a product.** Sessions live in memory, there is no
auth, and it runs Flask in debug mode. It exists to show the API surface a
real deployment shell needs. Production servers (auth, multi-tenancy, job
queues, persistence) belong in their own repo, depending on the published
`agentstan` package — exactly like this app does.

## Run

```bash
pip install agentstan flask flask-cors python-dotenv
python apps/server/app.py --port 5000
```

Set `OPENAI_API_KEY` (or pass `api_key` per request) for the `/api/generate`
and `/api/interpret` endpoints.

## Endpoints

| Method | Path | What |
|---|---|---|
| GET | `/api/health` | Version check |
| POST | `/api/generate` | `{prompt}` → validated spec (chat-to-ABM) |
| POST | `/api/simulate` | `{spec, steps}` → `{session_id}`; poll for progress |
| GET | `/api/simulate/<id>` | Run status + population history |
| POST | `/api/simulate/<id>/intervene` | Add/remove agents, change environment mid-run |
| POST | `/api/batch` | `{spec, n_runs, vary}` → batch results |
| POST | `/api/analyze` | Results → population/event analysis |
| POST | `/api/interpret` | Results → LLM explanation |
| POST | `/api/simulate/<id>/save`, `/api/simulate/load` | Checkpointing |
