from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
	id TEXT PRIMARY KEY,
	source TEXT NOT NULL,
	departure TEXT,
	destination TEXT,
	cargo_weight REAL,
	pay REAL,
	expires_at TEXT,
	data_json TEXT NOT NULL,
	fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_dep_dest ON jobs(departure, destination);
"""


@contextmanager
def connect(db_path: str):
	conn = sqlite3.connect(db_path)
	try:
		yield conn
	finally:
		conn.close()


def init_db(db_path: str) -> None:
	with connect(db_path) as conn:
		conn.executescript(_SCHEMA)
		conn.commit()


def _safe_get(obj: Dict[str, Any], *keys: str) -> Optional[Any]:
	cur: Any = obj
	for k in keys:
		if not isinstance(cur, dict) or k not in cur:
			return None
		cur = cur[k]
	return cur


def upsert_jobs(db_path: str, jobs: Iterable[Dict[str, Any]], *, source: str) -> int:
	"""Insert or update jobs; stores raw JSON and some projected fields if present.

	Returns number of rows upserted.
	"""
	rows = 0
	fetched_at = datetime.now(timezone.utc).isoformat()
	with connect(db_path) as conn:
		cursor = conn.cursor()
		for job in jobs:
			job_id = _safe_get(job, "Id") or _safe_get(job, "id") or _safe_get(job, "JobId")
			if not job_id:
				# Create a synthetic ID if absolutely needed
				job_id = json.dumps(job, sort_keys=True)
			departure = _safe_get(job, "Departure") or _safe_get(job, "departure")
			destination = _safe_get(job, "Destination") or _safe_get(job, "destination")
			cargo_weight = _safe_get(job, "CargoWeight") or _safe_get(job, "cargo_weight") or _safe_get(job, "Cargo")
			pay = _safe_get(job, "Pay") or _safe_get(job, "pay") or _safe_get(job, "Revenue")
			expires_at = _safe_get(job, "Expiration") or _safe_get(job, "expiration") or _safe_get(job, "ExpiresAt")

			cursor.execute(
				"""
				INSERT INTO jobs (id, source, departure, destination, cargo_weight, pay, expires_at, data_json, fetched_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(id) DO UPDATE SET
					source=excluded.source,
					departure=excluded.departure,
					destination=excluded.destination,
					cargo_weight=excluded.cargo_weight,
					pay=excluded.pay,
					expires_at=excluded.expires_at,
					data_json=excluded.data_json,
					fetched_at=excluded.fetched_at
				""",
				(
					job_id,
					source,
					departure,
					destination,
					float(cargo_weight) if isinstance(cargo_weight, (int, float)) else None,
					float(pay) if isinstance(pay, (int, float)) else None,
					str(expires_at) if expires_at is not None else None,
					json.dumps(job, ensure_ascii=False),
					fetched_at,
				),
			)
			rows += 1
		conn.commit()
	return rows
