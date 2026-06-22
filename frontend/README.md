# IndustryIQ — frontend

The web client for the IndustryIQ API: a dark, GPT-style chat app for
questioning your industry reports. React + Vite + TypeScript, styled with
Tailwind CSS v4. Self-contained package, separate from the Python backend.

## Features

- **Email auth** — register / log in; the JWT is stored in `localStorage` and
  sent as `Authorization: Bearer <token>` on every request.
- **Conversation sidebar** — list, create, rename, and delete your chats
  (scoped per account; you only see your own).
- **Streaming answers** — replies stream in token-by-token, with a step timeline
  (`Thinking → Searching → Generating`) where **each step shows a live timer**
  that freezes on completion, plus the server's total time.
- **Sources** — retrieved chunks shown per answer, with document name and score.
- **Document upload** — drag a `.pdf` / `.docx` / `.txt` onto the chat, or use
  the paperclip, to ground answers in that session's files.

## Run it

The backend must be running first (see the repo root `README.md`):

```powershell
# in another terminal, from the repo root:
python -m uvicorn industryiq.api.app:app --reload
```

Then, in this folder:

```powershell
npm install        # first time only
npm run dev        # starts the dev server at http://localhost:5173
```

Open <http://localhost:5173>, register an email, and start chatting.

## Configuration

The API base URL defaults to `http://localhost:8000`. To point at a different
backend, copy `.env.example` to `.env` and set `VITE_API_URL`:

```powershell
Copy-Item .env.example .env   # then edit VITE_API_URL
```

The backend must allow this origin via CORS (`CORS_ORIGINS`, default
`http://localhost:5173`).

## Build

```powershell
npm run build      # type-check (tsc) + production build into dist/
npm run preview    # serve the built app locally
```

## Structure

- `src/api.ts` — typed API client: auth, conversations, history, uploads, and
  the POST + Server-Sent-Events stream reader (uses `fetch` + `ReadableStream`,
  since `EventSource` is GET-only).
- `src/auth/AuthContext.tsx` — token + current-user state; routes a 401 to login.
- `src/components/` — `AuthScreen`, `Sidebar`, `ChatView`, `StepTimeline`,
  `Composer`, and friends.
- `src/index.css` — Tailwind v4 setup and the IndustryIQ theme (`@theme` tokens).
