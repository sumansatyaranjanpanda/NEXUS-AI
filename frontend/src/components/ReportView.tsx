import { useRef } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { EvalResult, PipelineStatusResponse } from "../api/types"

const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
}

interface Props {
  status: PipelineStatusResponse
  evalResult: EvalResult | null
  evalLoading: boolean
  onEval: () => void
  onReset: () => void
}

export function ReportView({ status, evalResult, evalLoading, onEval, onReset }: Props) {
  const reportRef = useRef<HTMLDivElement>(null)

  function handleDownloadPdf() {
    const content = reportRef.current?.innerHTML
    if (!content) return

    const win = window.open("", "_blank", "width=900,height=700")
    if (!win) return

    win.document.write(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NEXUS Report — ${status.query.slice(0, 80)}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; line-height: 1.7; color: #111; max-width: 760px; margin: 0 auto; padding: 2rem; }
    .report-meta { border-bottom: 2px solid #111; padding-bottom: 1rem; margin-bottom: 2rem; }
    .report-meta h1 { font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.4rem; }
    .report-meta p { font-size: 0.8rem; color: #555; margin: 0.15rem 0 0; }
    h1 { font-size: 1.6rem; margin: 0 0 1rem; line-height: 1.2; }
    h2 { font-size: 1.15rem; margin: 2rem 0 0.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
    h3 { font-size: 1rem; margin: 1.5rem 0 0.35rem; }
    p { margin: 0 0 0.85rem; }
    ul, ol { padding-left: 1.4rem; margin: 0 0 0.85rem; }
    li { margin-bottom: 0.25rem; }
    strong { font-weight: 600; }
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.88rem; }
    th, td { border: 1px solid #ddd; padding: 0.4rem 0.7rem; text-align: left; }
    th { background: #f6f6f6; font-weight: 600; }
    code { font-family: 'Courier New', monospace; font-size: 0.85em; background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 3px; }
    blockquote { border-left: 3px solid #ccc; margin: 0 0 1rem; padding-left: 1rem; color: #555; }
    @media print { body { padding: 0; } @page { margin: 2cm; } }
  </style>
</head>
<body>
  <div class="report-meta">
    <h1>NEXUS Competitive Intelligence Report</h1>
    <p>${status.query}</p>
    <p>Run: ${status.run_id} &nbsp;·&nbsp; ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</p>
  </div>
  ${content}
</body>
</html>`)
    win.document.close()
    // Small delay lets the new window finish rendering before the print dialog opens
    setTimeout(() => win.print(), 300)
  }

  return (
    <section className="report-view">
      <div className="report-header">
        <div>
          <h2>Report complete</h2>
          <code className="run-id">run: {status.run_id}</code>
        </div>
        <div className="report-header__actions">
          <button className="btn btn--secondary" onClick={handleDownloadPdf}>
            ↓ Download PDF
          </button>
          <button className="btn btn--secondary" onClick={onReset}>
            New query
          </button>
        </div>
      </div>

      <div className="card report-markdown" ref={reportRef}>
        <Markdown remarkPlugins={[remarkGfm]}>{status.report_markdown}</Markdown>
      </div>

      <div className="card">
        <div className="eval-header">
          <div>
            <strong>RAGAS quality evaluation</strong>
            <span className="hint">Measures faithfulness, relevancy, and context precision</span>
          </div>
          {!evalResult && (
            <button className="btn btn--primary" disabled={evalLoading} onClick={onEval}>
              {evalLoading ? "Evaluating…" : "Run evaluation"}
            </button>
          )}
        </div>

        {evalResult && (
          <div className="scores-grid">
            {Object.entries(evalResult.scores).map(([key, score]) => (
              <div key={key} className="score-card">
                <span className="score-value">
                  {score === -1 ? "—" : `${(score * 100).toFixed(0)}%`}
                </span>
                <span className="score-label">
                  {METRIC_LABELS[key] ?? key}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
