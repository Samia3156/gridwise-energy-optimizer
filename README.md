# GridWise Energy Optimizer

BUP CSE Fest 2026 Preliminary Hackathon project.

This repository contains a **Python + FastAPI HTTP API service** that:

1. Receives grid data and operator notes.
2. Uses an **LLM** to interpret every operator note (no hard-coded phrase matching).
3. Runs the LLM output through **deterministic guardrails / validation**.
4. Returns a valid **24-hour energy plan**.

This is the **initial scaffold**. Only `/health` and a placeholder `/optimize-energy` are implemented.

---

## Project layout

```
gridwise-energy-optimizer/
├── app/
│   └── main.py            # FastAPI entry point (routes live here for now)
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables (copy to .env)
├── .gitignore             # Files Git should ignore
└── README.md              # This file
```

---

## Prerequisites (Windows, beginner-friendly)

1. **Python 3.10 or newer**
   - Download: <https://www.python.org/downloads/windows/>
   - During install, **tick** "Add python.exe to PATH".
   - Verify in PowerShell:
     ```powershell
     py --version
     ```

2. **A code editor**
   - VS Code is recommended: <https://code.visualstudio.com/>

---

## Setup (one-time)

Open PowerShell in the project folder (`gridwise-energy-optimizer`) and run these commands **one by one**.

### 1. Create a virtual environment

```powershell
py -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

If PowerShell blocks the script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create your local `.env` file (only when you need API keys)

```powershell
Copy-Item .env.example .env
```

Edit `.env` later when the LLM integration is added. **Never commit `.env`.**

---

## Run the server

From the project folder, with the virtual environment active:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

You should see something like:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Leave this terminal open — it is the running server.

---

## Test the endpoints

Open a **second** PowerShell window (the first one is still running the server).

### Test `/health`

```powershell
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Test `/optimize-energy` (placeholder)

```powershell
curl -X POST http://127.0.0.1:8000/optimize-energy `
  -H "Content-Type: application/json" `
  -d "{\"operator_note\":\"shift load to off-peak hours\",\"grid_snapshot\":{\"load_mw\":[10]}}"
```

Expected response (a clearly marked placeholder):

```json
{
  "status": "not_implemented",
  "message": "Placeholder endpoint. ...",
  "received_payload_keys": ["operator_note", "grid_snapshot"]
}
```

You can also open the auto-generated API docs in a browser:

- <http://127.0.0.1:8000/docs>

---

## What is implemented vs. TODO

| Feature                              | Status |
| ------------------------------------ | ------ |
| `GET /health`                        | ✅ Done |
| `POST /optimize-energy` placeholder  | ✅ Done |
| Request/response schema (final)      | ⏳ TODO |
| LLM-based operator note interpreter  | ⏳ TODO |
| Deterministic guardrails / validation| ⏳ TODO |
| 24-hour optimizer                    | ⏳ TODO |
| Secrets via environment variables    | ✅ Planned via `.env` |

---

## Next steps (after this scaffold)

1. Define the exact request/response Pydantic schemas in `app/schemas.py`.
2. Add `app/llm.py` that calls the LLM provider using `LLM_API_KEY` from `.env`.
3. Add `app/guardrails.py` to validate the LLM output deterministically.
4. Add `app/optimizer.py` to compute the 24-hour plan.
5. Wire everything in `app/main.py`.

Do **not** hard-code API keys in source files.
