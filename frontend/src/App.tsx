import { useState } from "react"
import {
  ApiError,
  approvePipelineAsync,
  evaluatePipelineRun,
  getPipelineStatus,
  startPipelineAsync,
} from "./api/client"
import type { EvalResult, PipelineStatusResponse } from "./api/types"
import { HITLReview } from "./components/HITLReview"
import { KbView } from "./components/KbView"
import { PipelineProgress } from "./components/PipelineProgress"
import { QueryForm } from "./components/QueryForm"
import { ReportView } from "./components/ReportView"
import "./App.css"

type View = "idle" | "loading" | "hitl" | "complete" | "failed"
type Tab = "pipeline" | "kb"

export default function App() {
  const [tab, setTab] = useState<Tab>("pipeline")
  const [view, setView] = useState<View>("idle")

  // run_id is set immediately when startPipelineAsync() returns (~100ms).
  // PipelineProgress opens an SSE connection once run_id is available.
  const [runId, setRunId] = useState<string | null>(null)

  // reconnectToken: incrementing this value forces PipelineProgress to remount,
  // which closes the old SSE connection and opens a fresh one.
  // Used after HITL approval to reconnect for the remaining pipeline stages.
  const [reconnectToken, setReconnectToken] = useState(0)

  const [status, setStatus] = useState<PipelineStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [evalLoading, setEvalLoading] = useState(false)

  // ── Pipeline start ──────────────────────────────────────────────────────────

  async function handleQuerySubmit(query: string) {
    setView("loading")
    setError(null)
    setStatus(null)
    setEvalResult(null)
    setRunId(null)
    setReconnectToken(0)

    try {
      // Async dispatch: returns run_id in ~100ms.
      // PipelineProgress opens SSE on this run_id for live stage updates.
      const asyncResult = await startPipelineAsync(query)
      setRunId(asyncResult.run_id)
    } catch (err) {
      setError(err instanceof ApiError ? `${err.status}: ${err.message}` : String(err))
      setView("failed")
    }
  }

  // ── SSE callbacks — called by PipelineProgress when stages arrive ───────────

  async function handleHitl() {
    // HITL interrupt: fetch full state for the review screen
    if (!runId) return
    try {
      const fullStatus = await getPipelineStatus(runId)
      setStatus(fullStatus)
      setView("hitl")
    } catch {
      setError("Failed to load pipeline state for review.")
      setView("failed")
    }
  }

  async function handleComplete() {
    // Pipeline finished: fetch full state including report_markdown
    if (!runId) return
    try {
      const fullStatus = await getPipelineStatus(runId)
      setStatus(fullStatus)
      setView("complete")
    } catch {
      setError("Failed to load completed pipeline state.")
      setView("failed")
    }
  }

  function handleFailed(errors: string[]) {
    setError(errors.join("\n") || "Pipeline failed at an unexpected stage.")
    setView("failed")
  }

  // ── HITL approval ───────────────────────────────────────────────────────────

  async function handleApproval(approved: boolean) {
    if (!status) return
    setView("loading")
    setError(null)

    try {
      // Dispatch resume task to Celery (async). The pipeline state lives in Redis
      // so both the Celery worker and FastAPI can access it across process boundaries.
      await approvePipelineAsync(status.run_id, approved)

      if (!approved) {
        setError("Pipeline was rejected by human reviewer.")
        setView("failed")
        return
      }

      // Increment reconnectToken: forces PipelineProgress to remount with a fresh SSE
      // connection. The new SSE will detect the resumed pipeline and fire onComplete().
      setReconnectToken((t) => t + 1)
    } catch (err) {
      setError(err instanceof ApiError ? `${err.status}: ${err.message}` : String(err))
      setView("failed")
    }
  }

  // ── RAGAS evaluation ────────────────────────────────────────────────────────

  async function handleEval() {
    if (!status) return
    setEvalLoading(true)
    try {
      const result = await evaluatePipelineRun(status.run_id)
      setEvalResult(result)
    } catch {
      // Don't crash the report view — surface a placeholder score
      setEvalResult({ run_id: status.run_id, scores: { _error: -1 } })
    } finally {
      setEvalLoading(false)
    }
  }

  function handleReset() {
    setView("idle")
    setRunId(null)
    setStatus(null)
    setError(null)
    setEvalResult(null)
  }

  return (
    <main className="app">
      <header className="app-header">
        <h1>NEXUS</h1>
        <p>Autonomous competitive intelligence pipeline</p>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-btn${tab === "pipeline" ? " tab-btn--active" : ""}`}
          onClick={() => setTab("pipeline")}
        >
          Pipeline
        </button>
        <button
          className={`tab-btn${tab === "kb" ? " tab-btn--active" : ""}`}
          onClick={() => setTab("kb")}
        >
          Knowledge Base
        </button>
      </nav>

      {tab === "pipeline" && (
        <>
          {view === "idle" && <QueryForm onSubmit={handleQuerySubmit} />}

          {view === "loading" && (
            // key={reconnectToken} forces remount when token changes,
            // which closes the old SSE and opens a new one.
            <PipelineProgress
              key={reconnectToken}
              runId={runId}
              onHitl={handleHitl}
              onComplete={handleComplete}
              onFailed={handleFailed}
            />
          )}

          {view === "hitl" && status && (
            <HITLReview
              status={status}
              onApprove={() => handleApproval(true)}
              onReject={() => handleApproval(false)}
            />
          )}

          {view === "complete" && status && (
            <ReportView
              status={status}
              evalResult={evalResult}
              evalLoading={evalLoading}
              onEval={handleEval}
              onReset={handleReset}
            />
          )}

          {view === "failed" && (
            <section className="failed-view">
              <div className="failed-card">
                <h2>Pipeline failed</h2>
                <pre className="failed-message">{error ?? "An unknown error occurred."}</pre>
                <button className="btn btn--secondary" onClick={handleReset}>
                  Start over
                </button>
              </div>
            </section>
          )}
        </>
      )}

      {tab === "kb" && <KbView />}
    </main>
  )
}
