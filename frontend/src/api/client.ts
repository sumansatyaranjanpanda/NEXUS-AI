import type {
  AsyncRunResponse,
  EvalResult,
  GoldenEvalResult,
  IngestDocument,
  IngestResponse,
  KbStats,
  PipelineRequest,
  PipelineStatusResponse,
  SseEvent,
} from "./types"

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T
  }
  let detail = response.statusText
  try {
    const body = (await response.json()) as { detail?: string }
    if (body.detail) detail = body.detail
  } catch {
    // body wasn't JSON — fall back to statusText
  }
  throw new ApiError(response.status, detail)
}

// ── Pipeline (sync) ─────────────────────────────────────────────────────────
// Used as fallback when Celery worker is not running.

export async function startPipelineRun(query: string): Promise<PipelineStatusResponse> {
  const body: PipelineRequest = { query }
  const response = await fetch("/v1/reports/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  return parseOrThrow<PipelineStatusResponse>(response)
}

export async function approvePipelineRun(runId: string, approved: boolean): Promise<PipelineStatusResponse> {
  const response = await fetch("/v1/reports/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, approved }),
  })
  return parseOrThrow<PipelineStatusResponse>(response)
}

// ── Pipeline (async via Celery + SSE) ───────────────────────────────────────
// Requires Celery worker + redis-stack.

export async function startPipelineAsync(query: string): Promise<AsyncRunResponse> {
  const response = await fetch("/v1/reports/run/async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })
  return parseOrThrow<AsyncRunResponse>(response)
}

export async function approvePipelineAsync(runId: string, approved: boolean): Promise<void> {
  const response = await fetch("/v1/reports/approve/async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, approved }),
  })
  await parseOrThrow<{ run_id: string }>(response)
}

/**
 * Open an SSE connection for real-time pipeline stage updates.
 *
 * Returns a cleanup function — call it to close the connection.
 *
 * Why EventSource and not fetch + ReadableStream?
 * EventSource is the browser's built-in SSE client. It handles reconnection,
 * parsing, and memory management automatically. For one-way server→browser
 * streaming, it's the right primitive (WebSockets add bidirectional complexity
 * we don't need).
 */
export function createSseConnection(
  runId: string,
  onEvent: (event: SseEvent) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/v1/reports/stream/${runId}`)

  source.onmessage = (e) => {
    const data = JSON.parse(e.data) as SseEvent & { event?: string }
    if (data.event === "stream_end") {
      onEnd()
      source.close()
    } else {
      onEvent(data)
    }
  }

  source.onerror = () => {
    source.close()
  }

  return () => source.close()
}

// ── Status ───────────────────────────────────────────────────────────────────

export async function getPipelineStatus(runId: string): Promise<PipelineStatusResponse> {
  const response = await fetch(`/v1/reports/status/${runId}`)
  return parseOrThrow<PipelineStatusResponse>(response)
}

export async function evaluatePipelineRun(runId: string): Promise<EvalResult> {
  const response = await fetch(`/v1/reports/status/${runId}/evaluate`, {
    method: "POST",
  })
  return parseOrThrow<EvalResult>(response)
}

// ── Knowledge Base ───────────────────────────────────────────────────────────

export async function ingestDocuments(
  documents: IngestDocument[],
  sourceLabel = "manual-ingest",
): Promise<IngestResponse> {
  const response = await fetch("/v1/kb/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documents, source_label: sourceLabel }),
  })
  return parseOrThrow<IngestResponse>(response)
}

export async function getKbStats(): Promise<KbStats> {
  const response = await fetch("/v1/kb/stats")
  return parseOrThrow<KbStats>(response)
}

// ── Evaluation ───────────────────────────────────────────────────────────────

export async function evaluateGoldenDataset(fullPipeline = false): Promise<GoldenEvalResult> {
  const response = await fetch(`/v1/reports/evaluate/golden?full_pipeline=${fullPipeline}`, {
    method: "POST",
  })
  return parseOrThrow<GoldenEvalResult>(response)
}
