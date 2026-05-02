export interface PipelineRequest {
  query: string
}

export type PipelineStage =
  | "received"
  | "researching"
  | "building_kb"
  | "analyzing"
  | "fact_checking"
  | "hitl_review"
  | "complete"
  | "failed"

export interface PipelineStatusResponse {
  run_id: string
  query: string
  stage: PipelineStage
  analysis_summary: string
  fact_check_results: string[]
  report_markdown: string
  errors: string[]
  is_waiting_for_approval: boolean
}

export interface EvalResult {
  run_id: string
  scores: Record<string, number>
}

// Returned by POST /run/async — immediate, no blocking
export interface AsyncRunResponse {
  run_id: string
  task_id: string
  status: string
  poll_url: string
  stream_url: string
}

// A single progress log entry pushed by an agent during execution
export interface SseLogEntry {
  msg: string
  stage: string  // which agent stage emitted this (e.g. "researching", "analyzing")
  ts: number
}

// Each SSE event emitted by GET /stream/{run_id}
export interface SseEvent {
  run_id: string
  stage: PipelineStage
  is_waiting_for_approval: boolean
  analysis_summary: string
  errors: string[]
  logs?: SseLogEntry[]  // new progress messages since last poll
  event?: string  // "stream_end" when the server closes the connection
}

// POST /kb/ingest
export interface IngestDocument {
  text: string
  title: string
  url: string
}

export interface IngestResponse {
  source_label: string
  num_documents: number
  num_chunks: number
  num_upserted: number
}

// GET /kb/stats
export interface KbStats {
  collection: string
  points_count: number
  vectors_count: number
  indexed_vectors_count: number
  status: string
}

// POST /reports/evaluate/golden
export interface GoldenEvalResult {
  mode: "samples" | "pipeline"
  num_items: number
  avg_scores: Record<string, number>
  per_item_scores: Record<string, number>[]
}
