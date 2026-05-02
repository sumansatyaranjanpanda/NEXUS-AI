import { useState, type FormEvent } from "react"

const MIN_LENGTH = 3

const EXAMPLES = [
  "Compare Tesla vs Rivian on battery technology and production scale",
  "What are OpenAI's competitive advantages over Anthropic in enterprise?",
  "Analyze Stripe vs Braintree for developer-first payment processing",
]

interface Props {
  onSubmit: (query: string) => void
}

export function QueryForm({ onSubmit }: Props) {
  const [query, setQuery] = useState("")

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const trimmed = query.trim()
    if (trimmed.length >= MIN_LENGTH) onSubmit(trimmed)
  }

  return (
    <section className="query-form">
      <form onSubmit={handleSubmit}>
        <label htmlFor="query">Research query</label>
        <textarea
          id="query"
          rows={4}
          placeholder="e.g. How is Anthropic positioning Claude against OpenAI in enterprise sales?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={query.trim().length < MIN_LENGTH}>
          Start pipeline
        </button>
      </form>

      <div className="examples">
        <p className="examples-label">Examples</p>
        {EXAMPLES.map((ex) => (
          <button key={ex} className="example-btn" onClick={() => setQuery(ex)}>
            {ex}
          </button>
        ))}
      </div>
    </section>
  )
}
