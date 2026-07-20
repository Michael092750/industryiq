import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
} from "react";
import { useOutletContext, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import * as api from "../api";
import type { Source } from "../api";
import type { AppOutletContext } from "./AppLayout";
import { Composer } from "./Composer";
import { StepTimeline, type Step } from "./StepTimeline";
import { FileIcon } from "./icons";

interface ChatTurn {
  question: string;
  answer: string;
  sources?: Source[];
  steps?: Step[];
  totalMs?: number | null;
}

// The in-flight turn while the answer streams in.
interface Live {
  question: string;
  answer: string;
  steps: Step[];
  sources: Source[];
}

const PHASE_LABELS: Record<string, string> = {
  thinking: "Thinking",
  retrieving: "Searching knowledge base",
  broadening: "No filter matches — broadening search",
  generating: "Generating answer",
};

function closeLast(steps: Step[], now: number): Step[] {
  if (steps.length === 0) return steps;
  return steps.map((s, i) =>
    i === steps.length - 1 && s.doneMs === null ? { ...s, doneMs: now - s.startedAt } : s,
  );
}

function errorText(err: unknown): string {
  if (err instanceof api.ApiError) return err.message;
  return "Something went wrong. Please try again.";
}

export function ChatView() {
  const { id } = useParams();
  const { renameConversation } = useOutletContext<AppOutletContext>();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [live, setLive] = useState<Live | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);

  // Mirror `live` into a ref so the streaming `onDone` callback can read the
  // latest steps/sources without re-subscribing.
  const liveRef = useRef<Live | null>(null);
  useEffect(() => {
    liveRef.current = live;
  }, [live]);

  const scrollRef = useRef<HTMLDivElement>(null);

  // Load history + uploaded documents whenever the active conversation changes.
  useEffect(() => {
    if (!id) return;
    setLive(null);
    setError(null);
    setTurns([]);
    setDocuments([]);
    let cancelled = false;
    api
      .getHistory(id)
      .then((h) => {
        if (!cancelled) setTurns(h.map((t) => ({ question: t.question, answer: t.answer })));
      })
      .catch((e) => {
        if (!cancelled) setError(errorText(e));
      });
    api
      .listDocuments(id)
      .then((d) => {
        if (!cancelled) setDocuments(d);
      })
      .catch(() => {
        /* documents are best-effort */
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Keep the conversation pinned to the latest content.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight });
  }, [turns, live]);

  const busy = live !== null;

  const send = useCallback(
    async (question: string) => {
      if (!id || liveRef.current) return;
      setError(null);
      const wasEmpty = turns.length === 0;
      setLive({ question, answer: "", steps: [], sources: [] });
      await api.streamMessage(id, question, {
        onStatus: (phase) =>
          setLive((s) => {
            if (!s) return s;
            const now = performance.now();
            return {
              ...s,
              steps: [
                ...closeLast(s.steps, now),
                { key: phase, label: PHASE_LABELS[phase] ?? phase, startedAt: now, doneMs: null },
              ],
            };
          }),
        onSources: (_q, sources) => setLive((s) => (s ? { ...s, sources } : s)),
        onToken: (text) => setLive((s) => (s ? { ...s, answer: s.answer + text } : s)),
        onDone: (answer, timings) => {
          const cur = liveRef.current;
          const steps = closeLast(cur?.steps ?? [], performance.now());
          setTurns((prev) => [
            ...prev,
            { question, answer, sources: cur?.sources ?? [], steps, totalMs: timings.total ?? null },
          ]);
          setLive(null);
          if (wasEmpty) {
            const title =
              question.length > 48 ? `${question.slice(0, 48).trimEnd()}…` : question;
            renameConversation(id, title).catch(() => {});
          }
        },
        onError: (err) => {
          setError(errorText(err));
          setLive(null);
        },
      });
    },
    [id, turns.length, renameConversation],
  );

  const uploadFiles = useCallback(
    async (files: FileList) => {
      if (!id) return;
      for (const file of Array.from(files)) {
        try {
          await api.uploadDocument(id, file);
        } catch (e) {
          setError(errorText(e));
        }
      }
      try {
        setDocuments(await api.listDocuments(id));
      } catch {
        /* ignore */
      }
    },
    [id],
  );

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) void uploadFiles(e.dataTransfer.files);
  }

  if (!id) return null;

  return (
    <div
      className="relative flex h-full flex-col"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={onDrop}
    >
      {dragging && (
        <div className="pointer-events-none absolute inset-0 z-20 m-3 grid place-items-center rounded-2xl border-2 border-dashed border-accent/60 bg-accent/10 backdrop-blur-sm">
          <div className="text-center">
            <FileIcon width={28} height={28} className="mx-auto mb-2 text-accent-2" />
            <p className="text-sm font-medium text-white">Drop to add to this chat</p>
            <p className="text-xs text-fog">PDF, DOCX, or TXT</p>
          </div>
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-8">
          {turns.length === 0 && !live && <EmptyChat />}
          {turns.map((t, i) => (
            <TurnView key={i} turn={t} />
          ))}
          {live && <LiveView live={live} />}
          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-edge bg-ink/80 px-4 py-3 backdrop-blur">
        <div className="mx-auto max-w-3xl">
          {documents.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {documents.map((d) => (
                <span
                  key={d}
                  className="inline-flex items-center gap-1.5 rounded-full border border-edge bg-panel-2 px-2.5 py-1 text-xs text-fog"
                >
                  <FileIcon width={12} height={12} className="text-accent-2" />
                  {d}
                </span>
              ))}
            </div>
          )}
          <Composer disabled={busy} onSend={(t) => void send(t)} onFiles={(f) => void uploadFiles(f)} />
          <p className="mt-2 text-center text-[11px] text-fog-2">
            IndustryIQ can make mistakes. Verify important details.
          </p>
        </div>
      </div>
    </div>
  );
}

function TurnView({ turn }: { turn: ChatTurn }) {
  return (
    <div className="flex flex-col gap-4">
      <UserBubble text={turn.question} />
      <AssistantBubble>
        {turn.steps && turn.steps.length > 0 && (
          <StepTimeline steps={turn.steps} totalMs={turn.totalMs ?? null} />
        )}
        <Markdown text={turn.answer} />
        {turn.sources && turn.sources.length > 0 && <Sources sources={turn.sources} />}
      </AssistantBubble>
    </div>
  );
}

function LiveView({ live }: { live: Live }) {
  return (
    <div className="flex flex-col gap-4">
      <UserBubble text={live.question} />
      <AssistantBubble>
        <StepTimeline steps={live.steps} totalMs={null} />
        {live.answer ? (
          <Markdown text={live.answer} streaming />
        ) : (
          <span className="text-sm text-fog">Working on it…</span>
        )}
        {live.sources.length > 0 && <Sources sources={live.sources} />}
      </AssistantBubble>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm text-white">
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="iq-grad grid h-8 w-8 shrink-0 place-items-center rounded-lg text-[11px] font-bold text-white">
        IQ
      </div>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-edge bg-panel px-4 py-3 text-sm text-gray-100">
        {children}
      </div>
    </div>
  );
}

function Markdown({ text, streaming }: { text: string; streaming?: boolean }) {
  return (
    <div className="prose-iq">
      <ReactMarkdown>{text}</ReactMarkdown>
      {streaming && (
        <span
          className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-accent-2"
          style={{ animation: "blink 1s step-end infinite" }}
        />
      )}
    </div>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3 border-t border-edge pt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs font-medium text-fog transition hover:text-white"
      >
        {open ? "Hide" : "Show"} {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>
      {open && (
        <ul className="mt-2 flex flex-col gap-2">
          {sources.map((s, i) => (
            <li key={i} className="rounded-lg border border-edge bg-panel-2 p-2.5 text-xs text-fog">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="truncate font-medium text-accent-2">
                  {s.document ?? "Untitled source"}
                </span>
                <span className="shrink-0 tabular-nums text-fog-2">score {s.score.toFixed(2)}</span>
              </div>
              <p className="line-clamp-3">{s.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EmptyChat() {
  return (
    <div className="grid place-items-center py-24 text-center">
      <div className="iq-grad mb-4 grid h-14 w-14 place-items-center rounded-2xl text-lg font-bold text-white shadow-[0_0_30px_-6px_var(--color-accent)]">
        IQ
      </div>
      <h2 className="text-lg font-semibold text-white">Ask about your industry reports</h2>
      <p className="mt-1 max-w-sm text-sm text-fog">
        Type a question below, or drag in a PDF, DOCX, or TXT to ground the answer in your own
        document.
      </p>
    </div>
  );
}
