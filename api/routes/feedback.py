"""
Feedback API — collects user ratings and corrections on model responses.

Low-rated responses (rating < 3) are flagged for the augmentation pipeline.
All feedback is persisted to data/feedback/feedback_log.jsonl.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    """User feedback on a model response."""
    query: str = Field(..., description="The original query sent to the model")
    response: str = Field(..., description="The model's response")
    rating: int = Field(..., ge=1, le=5, description="User rating from 1 (poor) to 5 (excellent)")
    corrected_response: Optional[str] = Field(None, description="Optional corrected/ideal response")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for correlation")
    category: Optional[str] = Field(None, description="Response category (demand_forecast, mio_plan, etc.)")
    notes: Optional[str] = Field(None, description="Free-form user notes")


class FeedbackResponse(BaseModel):
    """Confirmation of feedback submission."""
    status: str
    feedback_id: str
    flagged_for_review: bool
    message: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback on a model response.
    
    Feedback is persisted to `data/feedback/feedback_log.jsonl`.
    Responses rated < 3 are flagged for augmentation pipeline review.
    If a corrected_response is provided with a low rating, it can be
    used directly as a training example for the next fine-tuning cycle.
    """
    settings = get_settings()

    # Generate feedback ID
    feedback_id = f"fb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    flagged = feedback.rating < 3

    # Build the record
    record = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": feedback.query,
        "response": feedback.response,
        "rating": feedback.rating,
        "corrected_response": feedback.corrected_response,
        "trace_id": feedback.trace_id,
        "category": feedback.category,
        "notes": feedback.notes,
        "flagged_for_review": flagged,
    }

    # Persist to JSONL
    feedback_dir = settings.data_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_dir / "feedback_log.jsonl"

    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Feedback %s saved (rating=%d, flagged=%s)",
            feedback_id, feedback.rating, flagged,
        )
    except Exception as e:
        logger.error("Failed to save feedback: %s", e)
        return FeedbackResponse(
            status="error",
            feedback_id=feedback_id,
            flagged_for_review=False,
            message=f"Failed to persist feedback: {e}",
        )

    # Score in Langfuse if trace_id provided
    if feedback.trace_id:
        try:
            from observability.langfuse import LangfuseTracker
            tracker = LangfuseTracker()
            client = tracker._get_client()
            if client:
                client.score(
                    trace_id=feedback.trace_id,
                    name="user_rating",
                    value=float(feedback.rating),
                    comment=feedback.notes or f"User rating: {feedback.rating}/5",
                )
        except Exception as e:
            logger.debug("Could not post feedback score to Langfuse: %s", e)

    message = "Feedback recorded successfully."
    if flagged:
        message += " Response flagged for augmentation pipeline review."
    if feedback.corrected_response:
        message += " Corrected response will be used for training data."

    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        flagged_for_review=flagged,
        message=message,
    )


@router.get("/feedback/stats")
async def feedback_stats() -> dict:
    """Return summary statistics of collected feedback."""
    settings = get_settings()
    feedback_file = settings.data_dir / "feedback" / "feedback_log.jsonl"

    if not feedback_file.exists():
        return {"total": 0, "avg_rating": 0.0, "flagged": 0, "with_corrections": 0}

    total = 0
    ratings = []
    flagged = 0
    with_corrections = 0

    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total += 1
                ratings.append(record.get("rating", 3))
                if record.get("flagged_for_review"):
                    flagged += 1
                if record.get("corrected_response"):
                    with_corrections += 1
    except Exception as e:
        logger.error("Error reading feedback stats: %s", e)

    return {
        "total": total,
        "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        "flagged": flagged,
        "with_corrections": with_corrections,
        "rating_distribution": {
            str(i): ratings.count(i) for i in range(1, 6)
        },
    }
