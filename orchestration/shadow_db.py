"""Shadow-mode DB writer for the forensics pipeline.

Writes PipelineResult flags and scores to the forensics_scores table
asynchronously — fire-and-forget, never blocks the user journey.

Usage (from your application layer)
------------------------------------
    from orchestration.shadow_db import write_async

    result = orchestration.run(image_bytes)
    write_async(application_id=app_id, client_id=client_id, result=result)
    # returns immediately; DB write happens in a background thread

DB setup
--------
Run the migration below once to create the table (adjust schema as needed):

    CREATE TABLE forensics_scores (
        id                          BIGSERIAL PRIMARY KEY,
        application_id              BIGINT       NOT NULL,
        client_id                   BIGINT,
        created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        model_version               TEXT         NOT NULL,
        -- Stage flags
        is_partial_document         BOOLEAN,
        is_screen_replay            BOOLEAN,
        is_printout                 BOOLEAN,
        is_tampered                 BOOLEAN,
        -- Scores
        screen_score                NUMERIC(6,4),
        -- Classification
        id_type_label               TEXT,
        label                       TEXT,
        risk_tier                   TEXT,
        -- Raw pipeline output (full JSON for audit / future replay)
        pipeline_json               JSONB
    );
    CREATE INDEX ON forensics_scores (application_id);
    CREATE INDEX ON forensics_scores (client_id);
    CREATE INDEX ON forensics_scores (created_at);

Environment variables required
-------------------------------
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD  (same as .env)

Note: this module is intentionally optional. If psycopg2 is not installed
or DB credentials are missing, all write calls silently no-op.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from typing import Optional

from orchestration.results import PipelineResult

logger = logging.getLogger(__name__)

MODEL_VERSION = os.getenv("FORENSICS_MODEL_VERSION", "v1.0")


def _get_connection():
    """Return a new psycopg2 connection using environment variables."""
    import psycopg2  # type: ignore
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def _to_json_safe(result: PipelineResult) -> str:
    """Serialise PipelineResult to a JSON string, stripping numpy arrays."""
    import numpy as np

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    return json.dumps(asdict(result), default=_default)


def _write_sync(
    application_id: int,
    client_id: Optional[int],
    result: PipelineResult,
    schema: str,
) -> None:
    """Synchronous DB write — called from background thread."""
    conn = None
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema}")
                screen_score = (
                    result.presentation_attack.screen_score
                    if result.presentation_attack else None
                )
                cur.execute(
                    """
                    INSERT INTO forensics_scores (
                        application_id, client_id, model_version,
                        is_partial_document, is_screen_replay, is_printout, is_tampered,
                        screen_score, id_type_label, label, risk_tier, pipeline_json
                    ) VALUES (
                        %(application_id)s, %(client_id)s, %(model_version)s,
                        %(is_partial_document)s, %(is_screen_replay)s,
                        %(is_printout)s, %(is_tampered)s,
                        %(screen_score)s, %(id_type_label)s, %(label)s,
                        %(risk_tier)s, %(pipeline_json)s
                    )
                    """,
                    {
                        "application_id": application_id,
                        "client_id": client_id,
                        "model_version": MODEL_VERSION,
                        "is_partial_document": result.is_partial_document,
                        "is_screen_replay": result.is_screen_replay,
                        "is_printout": result.is_printout,
                        "is_tampered": result.is_tampered,
                        "screen_score": float(screen_score) if screen_score is not None else None,
                        "id_type_label": result.id_type_label,
                        "label": result.label,
                        "risk_tier": result.risk_tier,
                        "pipeline_json": _to_json_safe(result),
                    },
                )
        logger.info(
            "forensics_scores written: application_id=%s risk_tier=%s",
            application_id,
            result.risk_tier,
        )
    except Exception as exc:
        # Shadow mode contract: log and swallow — never raise
        logger.error(
            "forensics_scores write failed (application_id=%s): %s",
            application_id,
            exc,
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def write_async(
    application_id: int,
    result: PipelineResult,
    client_id: Optional[int] = None,
    schema: str = "public",
) -> None:
    """Fire-and-forget: write pipeline result to DB in a daemon thread.

    Returns immediately. Failures are logged and never propagate.

    Args:
        application_id: ERP application primary key.
        result:         PipelineResult from orchestration.run().
        client_id:      Optional client PK for indexing.
        schema:         Postgres schema name (e.g. 'erp_finbro_ph').
    """
    if not os.environ.get("DB_HOST"):
        logger.debug("DB_HOST not set — skipping forensics_scores write")
        return

    t = threading.Thread(
        target=_write_sync,
        args=(application_id, client_id, result, schema),
        daemon=True,
        name=f"forensics-db-write-{application_id}",
    )
    t.start()
