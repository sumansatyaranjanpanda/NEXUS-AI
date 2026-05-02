import { useEffect, useState } from "react"
import { ApiError, evaluateGoldenDataset, getKbStats, ingestDocuments } from "../api/client"
import type { GoldenEvalResult, IngestResponse, KbStats } from "../api/types"

interface DocField {
  text: string
  title: string
  url: string
}

const EMPTY_DOC: DocField = { text: "", title: "", url: "" }

export function KbView() {
  const [docs, setDocs] = useState<DocField[]>([{ ...EMPTY_DOC }])
  const [sourceLabel, setSourceLabel] = useState("manual-ingest")
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null)
  const [ingestError, setIngestError] = useState<string | null>(null)

  const [stats, setStats] = useState<KbStats | null>(null)
  const [statsError, setStatsError] = useState(false)

  const [goldenLoading, setGoldenLoading] = useState(false)
  const [goldenResult, setGoldenResult] = useState<GoldenEvalResult | null>(null)
  const [goldenError, setGoldenError] = useState<string | null>(null)

  // Load KB stats on mount and after every successful ingest
  useEffect(() => {
    getKbStats()
      .then(setStats)
      .catch(() => setStatsError(true))
  }, [ingestResult])

  function addDoc() {
    setDocs((d) => [...d, { ...EMPTY_DOC }])
  }

  function removeDoc(i: number) {
    setDocs((d) => d.filter((_, idx) => idx !== i))
  }

  function updateDoc(i: number, field: keyof DocField, value: string) {
    setDocs((d) => d.map((doc, idx) => (idx === i ? { ...doc, [field]: value } : doc)))
  }

  async function handleIngest() {
    const valid = docs.filter((d) => d.text.trim().length >= 10)
    if (!valid.length) {
      setIngestError("Add at least one document with 10+ characters of text.")
      return
    }
    setIngestLoading(true)
    setIngestError(null)
    setIngestResult(null)
    try {
      const res = await ingestDocuments(valid, sourceLabel || "manual-ingest")
      setIngestResult(res)
      setDocs([{ ...EMPTY_DOC }])
    } catch (err) {
      setIngestError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setIngestLoading(false)
    }
  }

  async function handleGoldenEval() {
    setGoldenLoading(true)
    setGoldenError(null)
    setGoldenResult(null)
    try {
      const res = await evaluateGoldenDataset(false)  // samples mode (fast)
      setGoldenResult(res)
    } catch (err) {
      setGoldenError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setGoldenLoading(false)
    }
  }

  return (
    <div className="kb-view">
      {/* ── Stats ── */}
      <div className="card">
        <h2>Collection stats</h2>
        {statsError && <p className="hint">Could not reach Qdrant.</p>}
        {stats ? (
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-value">{stats.points_count ?? "—"}</span>
              <span className="stat-label">vectors stored</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{stats.status}</span>
              <span className="stat-label">collection status</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{stats.collection}</span>
              <span className="stat-label">collection name</span>
            </div>
          </div>
        ) : (
          !statsError && <p className="hint">Loading…</p>
        )}
      </div>

      {/* ── Ingest form ── */}
      <div className="card">
        <h2>Ingest documents</h2>
        <p className="hint" style={{ marginBottom: "1rem" }}>
          Pre-seed the knowledge base with company reports, articles, or competitor data.
          Better KB → better RAG retrieval → higher RAGAS scores.
        </p>

        <div className="ingest-meta">
          <label className="field-label">Source label</label>
          <input
            className="text-input"
            type="text"
            value={sourceLabel}
            onChange={(e) => setSourceLabel(e.target.value)}
            placeholder="e.g. tesla-q4-2023"
          />
        </div>

        {docs.map((doc, i) => (
          <div key={i} className="doc-block">
            <div className="doc-block-header">
              <span className="doc-block-num">Document {i + 1}</span>
              {docs.length > 1 && (
                <button
                  className="btn-remove"
                  onClick={() => removeDoc(i)}
                  aria-label="Remove document"
                >
                  ✕
                </button>
              )}
            </div>
            <label className="field-label">Title (optional)</label>
            <input
              className="text-input"
              type="text"
              value={doc.title}
              onChange={(e) => updateDoc(i, "title", e.target.value)}
              placeholder="Tesla Q4 2023 Press Release"
            />
            <label className="field-label">Source URL (optional)</label>
            <input
              className="text-input"
              type="text"
              value={doc.url}
              onChange={(e) => updateDoc(i, "url", e.target.value)}
              placeholder="https://ir.tesla.com/..."
            />
            <label className="field-label">Content *</label>
            <textarea
              className="doc-textarea"
              value={doc.text}
              onChange={(e) => updateDoc(i, "text", e.target.value)}
              placeholder="Paste the full document text here…"
              rows={6}
            />
          </div>
        ))}

        <div className="ingest-actions">
          <button className="btn btn--secondary" onClick={addDoc}>
            + Add document
          </button>
          <button className="btn btn--primary" disabled={ingestLoading} onClick={handleIngest}>
            {ingestLoading ? "Ingesting…" : "Ingest into KB"}
          </button>
        </div>

        {ingestError && <p className="inline-error">{ingestError}</p>}

        {ingestResult && (
          <div className="ingest-result">
            Ingested{" "}
            <strong style={{ color: "#4ade80" }}>{ingestResult.num_upserted} chunks</strong> from{" "}
            {ingestResult.num_documents} document(s) into{" "}
            <code>{ingestResult.source_label}</code>.
          </div>
        )}
      </div>

      {/* ── Golden benchmark ── */}
      <div className="card">
        <h2>Quality benchmark</h2>
        <p className="hint" style={{ marginBottom: "1rem" }}>
          Runs RAGAS <strong>answer_correctness</strong> against 3 curated Q&A pairs.
          Uses pre-written sample responses — no pipeline run needed (~30s).
        </p>
        <button
          className="btn btn--primary"
          disabled={goldenLoading}
          onClick={handleGoldenEval}
        >
          {goldenLoading ? "Running benchmark…" : "Run golden dataset eval"}
        </button>

        {goldenError && <p className="inline-error">{goldenError}</p>}

        {goldenResult && (
          <div className="golden-result">
            <p className="hint" style={{ marginBottom: "0.75rem" }}>
              Evaluated {goldenResult.num_items} items in <em>{goldenResult.mode}</em> mode
            </p>
            <div className="scores-grid">
              {Object.entries(goldenResult.avg_scores).map(([key, score]) => (
                <div key={key} className="score-card">
                  <span className="score-value">
                    {typeof score === "number" ? `${(score * 100).toFixed(0)}%` : "—"}
                  </span>
                  <span className="score-label">{key.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
