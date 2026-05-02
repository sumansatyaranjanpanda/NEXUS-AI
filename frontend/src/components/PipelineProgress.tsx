import { useEffect, useRef, useState } from "react"
import { createSseConnection } from "../api/client"
import type { PipelineStage, SseEvent, SseLogEntry } from "../api/types"

// The 5 visible stages (we skip "received" and "complete" from the display list)
const STAGES: { key: PipelineStage; label: string; desc: string }[] = [
  { key: "researching", label: "Research agent", desc: "Tavily web search + LLM synthesis" },
  { key: "building_kb", label: "Knowledge base", desc: "Chunk → embed → store in Qdrant" },
  { key: "analyzing", label: "Analysis agent", desc: "Hybrid RAG retrieval + structured report" },
  { key: "fact_checking", label: "Fact-check agent", desc: "Verifying every claim" },
  { key: "hitl_review", label: "Human review gate", desc: "Pausing for your approval" },
]

// Ordered pipeline stages — used to compute "done / active / pending" for each row
const STAGE_ORDER: PipelineStage[] = [
  "received",
  "researching",
  "building_kb",
  "analyzing",
  "fact_checking",
  "hitl_review",
  "complete",
]

type StageStatus = "pending" | "active" | "done"

function getStageStatus(key: PipelineStage, current: PipelineStage | null): StageStatus {
  if (!current) return "pending"
  const cur = STAGE_ORDER.indexOf(current)
  const tgt = STAGE_ORDER.indexOf(key)
  if (tgt < cur) return "done"
  if (tgt === cur) return "active"
  return "pending"
}

interface Props {
  // runId is null for the first ~100ms while we await startPipelineAsync().
  // Once set, PipelineProgress opens the SSE connection.
  runId: string | null
  onHitl: () => void
  onComplete: () => void
  onFailed: (errors: string[]) => void
}

export function PipelineProgress({ runId, onHitl, onComplete, onFailed }: Props) {
  const [currentStage, setCurrentStage] = useState<PipelineStage | null>(null)
  // logs accumulated per stage key — shown as live sub-items under each stage row
  const [stageLogs, setStageLogs] = useState<Record<string, string[]>>({})
  // true if the SSE connection has been open > 15s with no events — Celery may be down
  const [connectionWarning, setConnectionWarning] = useState(false)

  // Tracks whether the very first SSE event on this mount had is_waiting_for_approval.
  // If it did, we're reconnecting after HITL approval — the pipeline is already paused
  // and we should NOT call onHitl() again. We wait for it to advance to complete/failed.
  const firstEventIsWaiting = useRef(false)
  const receivedFirstEvent = useRef(false)

  useEffect(() => {
    if (!runId) return

    firstEventIsWaiting.current = false
    receivedFirstEvent.current = false
    setStageLogs({})

    // If we get no events within 15 seconds, the Celery worker likely isn't running
    const warningTimer = setTimeout(() => setConnectionWarning(true), 15_000)

    const disconnect = createSseConnection(
      runId,
      (event: SseEvent) => {
        clearTimeout(warningTimer)
        setConnectionWarning(false)
        setCurrentStage(event.stage)

        // Accumulate progress logs into their originating stage bucket
        if (event.logs && event.logs.length > 0) {
          setStageLogs((prev) => {
            const next = { ...prev }
            for (const entry of event.logs as SseLogEntry[]) {
              const key = entry.stage || event.stage
              next[key] = [...(next[key] ?? []), entry.msg]
            }
            return next
          })
        }

        if (!receivedFirstEvent.current) {
          receivedFirstEvent.current = true
          firstEventIsWaiting.current = event.is_waiting_for_approval
        }

        if (event.is_waiting_for_approval) {
          // Only trigger HITL review if this is NOT a reconnect that landed mid-pause.
          // On reconnect after approval the first event is already is_waiting — skip it.
          if (!firstEventIsWaiting.current) {
            onHitl()
          }
        } else if (event.stage === "complete") {
          onComplete()
        } else if (event.stage === "failed") {
          onFailed(event.errors)
        }
      },
      () => {
        // stream_end — server closed. No action needed; terminal events are handled above.
        clearTimeout(warningTimer)
      },
    )

    return () => {
      clearTimeout(warningTimer)
      disconnect()
    }
  }, [runId])  // runId changing = new pipeline → reconnect

  return (
    <section className="pipeline-progress">
      <div className="pipeline-progress__header">
        <span className="spinner" aria-hidden="true" />
        <h2>Pipeline running</h2>
        {currentStage ? (
          <p className="hint">
            Current stage:{" "}
            <strong style={{ color: "var(--accent)" }}>
              {currentStage.replace(/_/g, " ")}
            </strong>
          </p>
        ) : (
          <p className="hint">Connecting to pipeline stream…</p>
        )}
      </div>

      {connectionWarning && (
        <div className="warning-banner">
          No pipeline events received. Is the Celery worker running?
          <br />
          <code>docker compose up celery-worker</code>
        </div>
      )}

      <ol className="stage-list">
        {STAGES.map((s) => {
          const status = getStageStatus(s.key, currentStage)
          const logs = stageLogs[s.key] ?? []
          return (
            <li key={s.key} className={`stage-item${status === "active" ? " stage-item--active" : ""}`}>
              <span className={`stage-dot stage-dot--${status}`} aria-hidden="true" />
              <div>
                <strong
                  className="stage-label"
                  style={{ color: status === "done" ? "#4ade80" : undefined }}
                >
                  {status === "active" ? "▶ " : status === "done" ? "✓ " : ""}
                  {s.label}
                </strong>
                <span className="stage-desc">{s.desc}</span>
                {logs.length > 0 && (
                  <ul className="stage-logs">
                    {logs.map((msg, i) => (
                      <li key={i} className="stage-log-item">→ {msg}</li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
