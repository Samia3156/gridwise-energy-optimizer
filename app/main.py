"""
GridWise Energy Optimizer - FastAPI entry point.

This is STEP 7 of the scaffold (official schema migration):
- GET  /health              -> simple liveness check
- POST /optimize-energy     -> validates the official hackathon request
                                (scenario_id + 24 hours + battery + 1..3
                                operator notes), sends every operator note
                                through the LLM interpreter (app/llm.py),
                                runs the results through deterministic
                                guardrails (app/guardrails.py) that enforce
                                the official directive shape
                                (hours: List[int]), feeds the validated
                                directives into the 24-hour optimizer
                                (app/optimizer.py), runs internal
                                validation, and returns the final response
                                in the official shape (scenario_id,
                                directive_interpretation, hourly_plan,
                                totals, peak_grid_kwh, plan_summary).

Run locally:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status

from app.guardrails import validate_interpretations
from app.llm import LLMAPIError, LLMClient, LLMConfigError, LLMJSONError, LLMTimeoutError
from app.optimizer import InfeasiblePlanError, optimize
from app.schemas import (
    DirectiveInterpretation,
    OptimizeRequest,
    OptimizeResponse,
)

log = logging.getLogger("gridwise.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# LLM client (lazy, so missing env vars only break the optimize endpoint,
# not /health).
# ---------------------------------------------------------------------------

_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        try:
            _llm_client = LLMClient()
        except LLMConfigError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM not configured: {exc}",
            ) from exc
    return _llm_client


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("GridWise Energy Optimizer starting up (version=%s).", app.version)
    yield
    log.info("GridWise Energy Optimizer shutting down.")


app = FastAPI(
    title="GridWise Energy Optimizer",
    version="0.5.0",
    description="BUP CSE Fest 2026 Preliminary Hackathon - HTTP API service.",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Simple health-check endpoint used by judges / monitoring."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(payload: OptimizeRequest) -> OptimizeResponse:
    """
    STEP 7 endpoint (official schema).

    Pipeline:
        1. Validate the request against the official hackathon schema
           (done by FastAPI via the OptimizeRequest model).
        2. Send every operator note through the LLM interpreter.
        3. Validate every LLM result with deterministic guardrails
           (official directive shape with hours: List[int]).
        4. Feed the validated directives into the 24-hour optimizer,
           which self-validates the resulting plan.
        5. Return the final OptimizeResponse in the official shape:
           scenario_id + directive_interpretation (with note_index and
           explanation) + 24 hourly_plan entries + totals +
           peak_grid_kwh + plan_summary.
    """
    client = _get_llm_client()

    # 1 + 2. LLM is the interpretation path. Any failure becomes an
    #         appropriate HTTP error code.
    try:
        raw_interpretations = client.interpret_notes(payload)
    except LLMConfigError as exc:
        log.error("LLM config error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM not configured: {exc}",
        ) from exc
    except LLMTimeoutError as exc:
        log.error("LLM timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM request timed out: {exc}",
        ) from exc
    except LLMAPIError as exc:
        log.error("LLM API error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        ) from exc
    except LLMJSONError as exc:
        log.error("LLM JSON error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM returned invalid output: {exc}",
        ) from exc

    # 3. Guardrails validate every LLM result before it reaches the optimizer.
    validated: list[DirectiveInterpretation] = validate_interpretations(
        raw_interpretations, payload
    )

    # 3b. Stamp the official `note_index` from the request order. The LLM
    #     does not emit this field; the response schema requires it.
    for idx, interp in enumerate(validated):
        if interp.note_index != idx:
            object.__setattr__(interp, "note_index", idx)

    # 4 + 5. Deterministic 24-hour optimization with self-validation.
    try:
        return optimize(payload, validated)
    except InfeasiblePlanError as exc:
        log.error("Infeasible plan: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No feasible 24-hour plan: {exc}",
        ) from exc
