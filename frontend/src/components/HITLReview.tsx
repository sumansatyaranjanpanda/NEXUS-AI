import { useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { PipelineStatusResponse } from "../api/types"

interface Props {
  status: PipelineStatusResponse
  onApprove: () => void
  onReject: () => void
}

type Assessment = "pass" | "caveats" | "fail" | null

function parseAssessment(lines: string[]): Assessment {
  const joined = lines.join(" ").toUpperCase()
  if (joined.includes("PASS_WITH_CAVEATS")) return "caveats"
  if (joined.includes("OVERALL: PASS") || joined.includes("VERDICT: PASS") || joined.includes("RESULT: PASS")) return "pass"
  if (joined.includes("FAIL")) return "fail"
  if (joined.includes("PASS")) return "pass"
  return null
}

export function HITLReview({ status, onApprove, onReject }: Props) {
  const [submitting, setSubmitting] = useState(false)

  function approve() {
    setSubmitting(true)
    onApprove()
  }

  function reject() {
    setSubmitting(true)
    onReject()
  }

  const analysisMd = status.analysis_summary?.trim() || "No analysis available."
  const factCheckMd = status.fact_check_results.join("\n")
  const assessment = parseAssessment(status.fact_check_results)

  const assessmentLabel =
    assessment === "pass" ? "PASS" :
    assessment === "caveats" ? "PASS WITH CAVEATS" :
    assessment === "fail" ? "FAIL" : null

  const assessmentClass =
    assessment === "pass" ? "review__assessment review__assessment--pass" :
    assessment === "caveats" ? "review__assessment review__assessment--caveats" :
    assessment === "fail" ? "review__assessment review__assessment--fail" : ""

  return (
    <section className="review">
      <div className="review__header">
        <div className="review__title-row">
          <span className="review__icon" aria-hidden="true">🔍</span>
          <div>
            <h2 className="review__title">Human review</h2>
            <p className="review__subtitle">
              Review the pipeline output below, then approve or reject.
            </p>
          </div>
        </div>
        <div className="review__chips">
          <span className="chip chip--done">Research ✓</span>
          <span className="chip chip--done">Analysis ✓</span>
          <span className="chip chip--done">Fact-check ✓</span>
        </div>
      </div>

      <div className="review__card">
        <div className="review__card-label">Analysis</div>
        <div className="review__prose">
          <Markdown remarkPlugins={[remarkGfm]}>{analysisMd}</Markdown>
        </div>
      </div>

      {status.fact_check_results.length > 0 && (
        <div className="review__card">
          <div className="review__card-label-row">
            <span className="review__card-label">Fact-check</span>
            {assessmentLabel && (
              <span className={assessmentClass}>{assessmentLabel}</span>
            )}
          </div>
          <div className="review__prose review__prose--factcheck">
            <Markdown remarkPlugins={[remarkGfm]}>{factCheckMd}</Markdown>
          </div>
        </div>
      )}

      {status.errors.length > 0 && (
        <div className="review__card review__card--warning">
          <div className="review__card-label">Warnings</div>
          <ul className="review__warning-list">
            {status.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="review__actions">
        <button
          className="review__btn-approve"
          disabled={submitting}
          onClick={approve}
        >
          {submitting ? "Processing…" : "Approve & generate report"}
        </button>
        <button
          className="review__btn-reject"
          disabled={submitting}
          onClick={reject}
        >
          Reject
        </button>
      </div>
    </section>
  )
}
