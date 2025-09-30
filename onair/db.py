from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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

CREATE TABLE IF NOT EXISTS airports (
	icao TEXT PRIMARY KEY,
	name TEXT,
	latitude REAL,
	longitude REAL,
	size INTEGER NOT NULL,
	country_code TEXT,
	updated_at TEXT NOT NULL,
	data_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS airplanes (
	id TEXT PRIMARY KEY,
	registration TEXT,
	type TEXT,
	model TEXT,
	icao TEXT,
	status TEXT,
	location_icao TEXT,
	latitude REAL,
	longitude REAL,
	fuel_total REAL,
	payload_capacity REAL,
	seats INTEGER,
	updated_at TEXT NOT NULL,
	data_json TEXT NOT NULL
);
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
	rows = 0
	fetched_at = datetime.now(timezone.utc).isoformat()
	with connect(db_path) as conn:
		cursor = conn.cursor()
		for job in jobs:
			job_id = _safe_get(job, "Id") or _safe_get(job, "id") or _safe_get(job, "JobId")
			if not job_id:
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


def upsert_airport(db_path: str, airport: Dict[str, Any]) -> None:
	icao = _safe_get(airport, "ICAO") or _safe_get(airport, "icao")
	name = _safe_get(airport, "Name") or _safe_get(airport, "name")
	lat = _safe_get(airport, "Latitude") or _safe_get(airport, "latitude")
	lon = _safe_get(airport, "Longitude") or _safe_get(airport, "longitude")
	size = _safe_get(airport, "Size") or _safe_get(airport, "size")
	country = _safe_get(airport, "CountryCode") or _safe_get(airport, "country_code")
	updated_at = datetime.now(timezone.utc).isoformat()
	with connect(db_path) as conn:
		conn.execute(
			"""
			INSERT INTO airports (icao, name, latitude, longitude, size, country_code, updated_at, data_json)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(icao) DO UPDATE SET
				name=excluded.name,
				latitude=excluded.latitude,
				longitude=excluded.longitude,
				size=excluded.size,
				country_code=excluded.country_code,
				updated_at=excluded.updated_at,
				data_json=excluded.data_json
			""",
			(
				icao,
				name,
				float(lat) if isinstance(lat, (int, float)) else None,
				float(lon) if isinstance(lon, (int, float)) else None,
				int(size) if isinstance(size, (int, float)) else 0,
				country,
				updated_at,
				json.dumps(airport, ensure_ascii=False),
			),
		)
		conn.commit()


def get_airport(db_path: str, icao: str) -> Optional[Dict[str, Any]]:
	with connect(db_path) as conn:
		row = conn.execute(
			"SELECT icao, name, latitude, longitude, size, country_code, updated_at, data_json FROM airports WHERE icao = ?",
			(icao,),
		).fetchone()
		if not row:
			return None
		return {
			"icao": row[0],
			"name": row[1],
			"latitude": row[2],
			"longitude": row[3],
			"size": row[4],
			"country_code": row[5],
			"updated_at": row[6],
			"data_json": row[7],
		}


def is_airport_stale(db_path: str, icao: str, max_age_days: int) -> bool:
	ap = get_airport(db_path, icao)
	if not ap:
		return True
	try:
		updated = datetime.fromisoformat(ap["updated_at"]).replace(tzinfo=timezone.utc)
	except Exception:
		return True
	return updated < datetime.now(timezone.utc) - timedelta(days=max_age_days)


def upsert_airplanes(db_path: str, airplanes: Iterable[Dict[str, Any]]) -> int:
	rows = 0
	updated_at = datetime.now(timezone.utc).isoformat()
	with connect(db_path) as conn:
		cur = conn.cursor()
		for ap in airplanes:
			ap_id = _safe_get(ap, "Id") or _safe_get(ap, "id")
			reg = _safe_get(ap, "Registration") or _safe_get(ap, "TailNumber") or _safe_get(ap, "registration")
			type_name = _safe_get(ap, "TypeName") or _safe_get(ap, "type")
			model = _safe_get(ap, "Model") or _safe_get(ap, "model")
			aircraft_icao = _safe_get(ap, "AircraftTypeICAO") or _safe_get(ap, "icao")
			status = _safe_get(ap, "State") or _safe_get(ap, "Status") or _safe_get(ap, "state")
			loc = _safe_get(ap, "CurrentAirportICAO") or _safe_get(ap, "LocationICAO") or _safe_get(ap, "location_icao")
			lat = _safe_get(ap, "Latitude") or _safe_get(ap, "latitude")
			lon = _safe_get(ap, "Longitude") or _safe_get(ap, "longitude")
			fuel_total = _safe_get(ap, "FuelTotalQuantity") or _safe_get(ap, "fuel_total")
			payload_cap = _safe_get(ap, "MaxPayload") or _safe_get(ap, "payload_capacity")
			seats = _safe_get(ap, "Seats") or _safe_get(ap, "seats")

			cur.execute(
				"""
				INSERT INTO airplanes (id, registration, type, model, icao, status, location_icao, latitude, longitude, fuel_total, payload_capacity, seats, updated_at, data_json)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(id) DO UPDATE SET
					registration=excluded.registration,
					type=excluded.type,
					model=excluded.model,
					icao=excluded.icao,
					status=excluded.status,
					location_icao=excluded.location_icao,
					latitude=excluded.latitude,
					longitude=excluded.longitude,
					fuel_total=excluded.fuel_total,
					payload_capacity=excluded.payload_capacity,
					seats=excluded.seats,
					updated_at=excluded.updated_at,
					data_json=excluded.data_json
				""",
				(
					ap_id,
					reg,
					type_name,
					model,
					aircraft_icao,
					status,
					loc,
					float(lat) if isinstance(lat, (int, float)) else None,
					float(lon) if isinstance(lon, (int, float)) else None,
					float(fuel_total) if isinstance(fuel_total, (int, float)) else None,
					float(payload_cap) if isinstance(payload_cap, (int, float)) else None,
					int(seats) if isinstance(seats, (int, float)) else None,
					updated_at,
					json.dumps(ap, ensure_ascii=False),
				),
			)
			rows += 1
		conn.commit()
	return rows
