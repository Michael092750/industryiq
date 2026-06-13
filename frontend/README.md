# frontend — ragproject UI

A React + Vite + TypeScript single-page app that talks to the ragproject API
(`/ingest`, `/query`). Self-contained package, separate from the Python backend.

## Run it

The backend must be running first (see the repo root README / DEPLOY.md):

```powershell
# in another terminal, from the repo root:
python -m uvicorn ragproject.api.app:app --reload
```

Then, in this folder:

```powershell
npm install        # first time only
npm run dev        # starts the dev server at http://localhost:5173
```

Open <http://localhost:5173>, paste some text to ingest, then ask a question.

## Configuration

The API base URL defaults to `http://localhost:8000`. To point at a different
backend, copy `.env.example` to `.env` and set `VITE_API_URL`.

The backend must allow this origin via CORS (`CORS_ORIGINS`, default
`http://localhost:5173`).

## Build

```powershell
npm run build      # type-check + production build into dist/
npm run preview    # serve the built app locally
```
