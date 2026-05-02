"""Unit tests for the RAGAS evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nexus.evaluation.ragas_eval import evaluate_state
from nexus.schemas.state import PipelineStage, create_initial_state


class TestRagasEval:
    """Tests for evaluate_state using RAGAS."""

    def test_missing_data_returns_none(self) -> None:
        """Should return None if state is missing query, context, or response."""
        state = create_initial_state("test")
        # Missing context and response
        assert evaluate_state(state) is None

    def test_evaluate_state_success(self) -> None:
        """Should return scores when evaluate succeeds."""
        state = create_initial_state("What is the capital of France?")
        state["retrieved_context"] = ["Paris is the capital of France."]
        state["analysis_summary"] = "The capital is Paris."

        mock_scores = {"faithfulness": 1.0, "answer_relevancy": 0.9, "context_precision": 1.0}

        with (
            patch("nexus.evaluation.ragas_eval.Settings"),
            patch("nexus.evaluation.ragas_eval.ChatOpenAI"),
            patch("nexus.evaluation.ragas_eval.FastEmbedEmbeddings"),
            patch("nexus.evaluation.ragas_eval.LangchainLLMWrapper"),
            patch("nexus.evaluation.ragas_eval.LangchainEmbeddingsWrapper"),
            patch("nexus.evaluation.ragas_eval.evaluate", return_value=mock_scores),
        ):
            result = evaluate_state(state)
        
        assert result is not None
        assert result["faithfulness"] == 1.0
        assert result["answer_relevancy"] == 0.9
