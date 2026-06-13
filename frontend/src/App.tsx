import { useState } from "react";
import { ingest, query, type QueryResult } from "./api";
import "./App.css";

function App() {
  const [docText, setDocText] = useState("");
  const [source, setSource] = useState("");
  const [ingestMsg, setIngestMsg] = useState("");

  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  async function handleIngest() {
    setError("");
    setIngestMsg("");
    try {
      const ids = await ingest(docText, source || undefined);
      setIngestMsg(`Indexed ${ids.length} chunk(s).`);
      setDocText("");
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleQuery() {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      setResult(await query(question));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>ragproject</h1>
        <p className="subtitle">Ask questions about your documents</p>
      </header>

      <section className="card">
        <h2>1. Add a document</h2>
        <textarea
          value={docText}
          onChange={(e) => setDocText(e.target.value)}
          placeholder="Paste some text to index..."
          rows={4}
        />
        <div className="row">
          <input
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="source name (optional)"
          />
          <button onClick={handleIngest} disabled={!docText.trim()}>
            Ingest
          </button>
        </div>
        {ingestMsg && <p className="ok">{ingestMsg}</p>}
      </section>

      <section className="card">
        <h2>2. Ask a question</h2>
        <div className="row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && question.trim() && handleQuery()}
            placeholder="What do you want to know?"
          />
          <button onClick={handleQuery} disabled={!question.trim() || loading}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>

        {result && (
          <div className="result">
            <h3>Answer</h3>
            <p className="answer">{result.answer}</p>
            {result.sources.length > 0 && (
              <>
                <h3>Sources</h3>
                <ul className="sources">
                  {result.sources.map((s, i) => (
                    <li key={i}>
                      <span className="score">{s.score.toFixed(2)}</span>
                      <span>{s.text}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </section>

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;
