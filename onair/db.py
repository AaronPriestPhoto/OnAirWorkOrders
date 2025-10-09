from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, List, Tuple


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
	id TEXT PRIMARY KEY,
	source TEXT NOT NULL,
	departure TEXT,
	destination TEXT,
	cargo_weight REAL,
	pay REAL,
	xp REAL,
	expires_at TEXT,
	computed_distance_nm REAL,
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

CREATE TABLE IF NOT EXISTS job_legs (
	job_id TEXT NOT NULL,
	leg_index INTEGER NOT NULL,
	from_icao TEXT NOT NULL,
	to_icao TEXT NOT NULL,
	from_lat REAL,
	from_lon REAL,
	to_lat REAL,
	to_lon REAL,
	distance_nm REAL,
	cargo_lbs REAL,
	PRIMARY KEY(job_id, leg_index)
);

CREATE TABLE IF NOT EXISTS plane_specs (
	plane_type TEXT PRIMARY KEY,
	cruise_speed_kts REAL,
	min_airport_size INTEGER,
	range1_nm REAL,
	payload1_lbs REAL,
	range2_nm REAL,
	payload2_lbs REAL,
	priority TEXT,
	fuel_type TEXT,
	updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_scores (
	plane_id TEXT NOT NULL,
	plane_type TEXT,
	job_id TEXT NOT NULL,
	feasible INTEGER NOT NULL,
	reason TEXT,
	pay_per_hour REAL,
	xp_per_hour REAL,
	balance_score REAL,
	PRIMARY KEY(plane_id, job_id)
);
"""


@contextmanager
def connect(db_path: str):
	conn = sqlite3.connect(db_path)
	try:
		# Speed-oriented pragmas (acceptable for local app DB)
		conn.execute("PRAGMA journal_mode=WAL")
		conn.execute("PRAGMA synchronous=NORMAL")
		conn.execute("PRAGMA temp_store=MEMORY")
		# Additional optimizations for reduced disk writes
		conn.execute("PRAGMA cache_size=10000")  # 10MB cache
		conn.execute("PRAGMA page_size=4096")    # Larger page size for better I/O
		conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
		yield conn
	finally:
		conn.close()


def init_db(db_path: str) -> None:
	with connect(db_path) as conn:
		conn.executescript(_SCHEMA)
		conn.commit()


def migrate_schema(db_path: str) -> None:
	with connect(db_path) as conn:
		c = conn.cursor()
		# jobs additions
		cols = [r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()]
		if "computed_distance_nm" not in cols:
			c.execute("ALTER TABLE jobs ADD COLUMN computed_distance_nm REAL")
		if "xp" not in cols:
			c.execute("ALTER TABLE jobs ADD COLUMN xp REAL")
		# job_legs table and columns
		c.execute(
			"""
			CREATE TABLE IF NOT EXISTS job_legs (
				job_id TEXT NOT NULL,
				leg_index INTEGER NOT NULL,
				from_icao TEXT NOT NULL,
				to_icao TEXT NOT NULL,
				from_lat REAL,
				from_lon REAL,
				to_lat REAL,
				to_lon REAL,
				distance_nm REAL,
				cargo_lbs REAL,
				PRIMARY KEY(job_id, leg_index)
			)
			"""
		)
		cols_legs = [r[1] for r in c.execute("PRAGMA table_info(job_legs)").fetchall()]
		if "cargo_lbs" not in cols_legs:
			c.execute("ALTER TABLE job_legs ADD COLUMN cargo_lbs REAL")
		# plane_specs
		c.execute(
			"""
			CREATE TABLE IF NOT EXISTS plane_specs (
				plane_type TEXT PRIMARY KEY,
				cruise_speed_kts REAL,
				min_airport_size INTEGER,
				range1_nm REAL,
				payload1_lbs REAL,
				range2_nm REAL,
				payload2_lbs REAL,
				priority TEXT,
				fuel_type TEXT,
				updated_at TEXT NOT NULL
			)
			"""
		)
		# job_scores
		c.execute(
			"""
			CREATE TABLE IF NOT EXISTS job_scores (
				plane_id TEXT NOT NULL,
				plane_type TEXT,
				job_id TEXT NOT NULL,
				feasible INTEGER NOT NULL,
				reason TEXT,
				pay_per_hour REAL,
				xp_per_hour REAL,
				balance_score REAL,
				PRIMARY KEY(plane_id, job_id)
			)
			"""
		)
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
		job_data = []
		for job in jobs:
			job_id = _safe_get(job, "Id") or _safe_get(job, "id") or _safe_get(job, "JobId")
			if not job_id:
				job_id = json.dumps(job, sort_keys=True)
			departure = _safe_get(job, "Departure") or _safe_get(job, "departure")
			destination = _safe_get(job, "Destination") or _safe_get(job, "destination")
			cargo_weight = _safe_get(job, "CargoWeight") or _safe_get(job, "cargo_weight") or _safe_get(job, "Cargo")
			pay = _safe_get(job, "RealPay") or _safe_get(job, "Pay") or _safe_get(job, "pay") or _safe_get(job, "Revenue")
			xp = _safe_get(job, "XP") or _safe_get(job, "xp")
			expires_at = _safe_get(job, "Expiration") or _safe_get(job, "expiration") or _safe_get(job, "ExpiresAt")

			job_data.append((
				job_id,
				source,
				departure,
				destination,
				float(cargo_weight) if isinstance(cargo_weight, (int, float)) else None,
				float(pay) if isinstance(pay, (int, float)) else None,
				float(xp) if isinstance(xp, (int, float)) else None,
				str(expires_at) if expires_at is not None else None,
				job_id,
				json.dumps(job, ensure_ascii=False),
				fetched_at,
			))
			rows += 1
		
		# Batch insert/update all jobs in a single transaction
		cursor.executemany(
			"""
			INSERT INTO jobs (id, source, departure, destination, cargo_weight, pay, xp, expires_at, computed_distance_nm, data_json, fetched_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT computed_distance_nm FROM jobs WHERE id=?), NULL), ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				source=excluded.source,
				departure=excluded.departure,
				destination=excluded.destination,
				cargo_weight=excluded.cargo_weight,
				pay=excluded.pay,
				xp=excluded.xp,
				expires_at=excluded.expires_at,
				data_json=excluded.data_json,
				fetched_at=excluded.fetched_at
			""",
			job_data
		)
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
		airplane_data = []
		for ap in airplanes:
			ap_id = _safe_get(ap, "Id") or _safe_get(ap, "id")
			reg = _safe_get(ap, "Registration") or _safe_get(ap, "TailNumber") or _safe_get(ap, "Ident") or _safe_get(ap, "Identifier") or _safe_get(ap, "registration")
			# Extract type from nested AircraftType object or direct fields
			aircraft_type_obj = ap.get("AircraftType", {})
			type_name = (
				_safe_get(aircraft_type_obj, "DisplayName") or 
				_safe_get(aircraft_type_obj, "TypeName") or
				_safe_get(ap, "TypeName") or 
				_safe_get(ap, "type")
			)
			model = (
				_safe_get(aircraft_type_obj, "TypeName") or
				_safe_get(ap, "Model") or 
				_safe_get(ap, "model")
			)
			aircraft_icao = _safe_get(ap, "AircraftTypeICAO") or _safe_get(ap, "icao")
			status = _safe_get(ap, "State") or _safe_get(ap, "Status") or _safe_get(ap, "state")
			# Try to get location from various possible fields, including nested CurrentAirport.ICAO
			loc = (
				_safe_get(ap, "CurrentAirportICAO") or 
				_safe_get(ap, "LocationICAO") or 
				_safe_get(ap, "location_icao") or
				_safe_get(ap.get("CurrentAirport", {}), "ICAO")
			)
			lat = _safe_get(ap, "Latitude") or _safe_get(ap, "latitude")
			lon = _safe_get(ap, "Longitude") or _safe_get(ap, "longitude")
			fuel_total = _safe_get(ap, "FuelTotalQuantity") or _safe_get(ap, "fuel_total")
			payload_cap = _safe_get(ap, "MaxPayload") or _safe_get(ap, "payload_capacity")
			seats = _safe_get(ap, "Seats") or _safe_get(ap, "seats")

			airplane_data.append((
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
			))
			rows += 1
		
		# Batch insert/update all airplanes in a single transaction
		cur.executemany(
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
			airplane_data
		)
		conn.commit()
	return rows


def clear_job_legs(db_path: str) -> None:
	with connect(db_path) as conn:
		conn.execute("DELETE FROM job_legs")
		conn.commit()


def upsert_job_leg(db_path: str, job_id: str, leg_index: int, from_icao: str, to_icao: str, from_lat: float, from_lon: float, to_lat: float, to_lon: float, distance_nm: float, cargo_lbs: Optional[float] = None) -> None:
	with connect(db_path) as conn:
		conn.execute(
			"""
			INSERT INTO job_legs (job_id, leg_index, from_icao, to_icao, from_lat, from_lon, to_lat, to_lon, distance_nm, cargo_lbs)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(job_id, leg_index) DO UPDATE SET
				from_icao=excluded.from_icao,
				to_icao=excluded.to_icao,
				from_lat=excluded.from_lat,
				from_lon=excluded.from_lon,
				to_lat=excluded.to_lat,
				to_lon=excluded.to_lon,
				distance_nm=excluded.distance_nm,
				cargo_lbs=excluded.cargo_lbs
			""",
			(job_id, leg_index, from_icao, to_icao, from_lat, from_lon, to_lat, to_lon, distance_nm, cargo_lbs),
		)
		conn.commit()


def upsert_job_legs_bulk(db_path: str, legs: Iterable[Tuple[str, int, str, str, float, float, float, float, float, Optional[float]]]) -> None:
	"""Insert or update many job legs in a single transaction."""
	with connect(db_path) as conn:
		conn.executemany(
			"""
			INSERT INTO job_legs (job_id, leg_index, from_icao, to_icao, from_lat, from_lon, to_lat, to_lon, distance_nm, cargo_lbs)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(job_id, leg_index) DO UPDATE SET
				from_icao=excluded.from_icao,
				to_icao=excluded.to_icao,
				from_lat=excluded.from_lat,
				from_lon=excluded.from_lon,
				to_lat=excluded.to_lat,
				to_lon=excluded.to_lon,
				distance_nm=excluded.distance_nm,
				cargo_lbs=excluded.cargo_lbs
			""",
			list(legs),
		)
		conn.commit()


def update_job_total_distance(db_path: str, job_id: str, total_nm: float) -> None:
	with connect(db_path) as conn:
		conn.execute("UPDATE jobs SET computed_distance_nm = ? WHERE id = ?", (total_nm, job_id))
		conn.commit()


def update_job_total_distances_bulk(db_path: str, distances: Iterable[Tuple[float, str]]) -> None:
	"""Update many job total distances in a single transaction."""
	with connect(db_path) as conn:
		conn.executemany(
			"UPDATE jobs SET computed_distance_nm = ? WHERE id = ?",
			list(distances)
		)
		conn.commit()


def upsert_plane_spec(db_path: str, *, plane_type: str, cruise_speed_kts: Optional[float], min_airport_size: Optional[int], range1_nm: Optional[float], payload1_lbs: Optional[float], range2_nm: Optional[float], payload2_lbs: Optional[float], priority: Optional[str], fuel_type: Optional[str]) -> None:
	from datetime import datetime, timezone
	updated_at = datetime.now(timezone.utc).isoformat()
	with connect(db_path) as conn:
		conn.execute(
			"""
			INSERT INTO plane_specs (plane_type, cruise_speed_kts, min_airport_size, range1_nm, payload1_lbs, range2_nm, payload2_lbs, priority, fuel_type, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(plane_type) DO UPDATE SET
				cruise_speed_kts=excluded.cruise_speed_kts,
				min_airport_size=excluded.min_airport_size,
				range1_nm=excluded.range1_nm,
				payload1_lbs=excluded.payload1_lbs,
				range2_nm=excluded.range2_nm,
				payload2_lbs=excluded.payload2_lbs,
				priority=excluded.priority,
				fuel_type=excluded.fuel_type,
				updated_at=excluded.updated_at
			""",
			(
				plane_type,
				float(cruise_speed_kts) if cruise_speed_kts is not None else None,
				int(min_airport_size) if min_airport_size is not None else None,
				float(range1_nm) if range1_nm is not None else None,
				float(payload1_lbs) if payload1_lbs is not None else None,
				float(range2_nm) if range2_nm is not None else None,
				float(payload2_lbs) if payload2_lbs is not None else None,
				priority,
				fuel_type,
				updated_at,
			),
		)
		conn.commit()


def clear_job_scores(db_path: str) -> None:
	with connect(db_path) as conn:
		conn.execute("DELETE FROM job_scores")
		conn.commit()


def upsert_job_score(db_path: str, *, plane_id: str, plane_type: Optional[str], job_id: str, feasible: bool, reason: Optional[str], pay_per_hour: Optional[float], xp_per_hour: Optional[float], balance_score: Optional[float]) -> None:
	with connect(db_path) as conn:
		conn.execute(
			"""
			INSERT INTO job_scores (plane_id, plane_type, job_id, feasible, reason, pay_per_hour, xp_per_hour, balance_score)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(plane_id, job_id) DO UPDATE SET
				plane_type=excluded.plane_type,
				feasible=excluded.feasible,
				reason=excluded.reason,
				pay_per_hour=excluded.pay_per_hour,
				xp_per_hour=excluded.xp_per_hour,
				balance_score=excluded.balance_score
			""",
			(plane_id, plane_type, job_id, 1 if feasible else 0, reason, pay_per_hour, xp_per_hour, balance_score),
		)
		conn.commit()


def upsert_job_scores_bulk(db_path: str, scores: Iterable[Tuple[str, Optional[str], str, int, Optional[str], Optional[float], Optional[float], Optional[float]]]) -> None:
	"""Insert or update many job_scores rows in a single transaction."""
	with connect(db_path) as conn:
		conn.executemany(
			"""
			INSERT INTO job_scores (plane_id, plane_type, job_id, feasible, reason, pay_per_hour, xp_per_hour, balance_score)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(plane_id, job_id) DO UPDATE SET
				plane_type=excluded.plane_type,
				feasible=excluded.feasible,
				reason=excluded.reason,
				pay_per_hour=excluded.pay_per_hour,
				xp_per_hour=excluded.xp_per_hour,
				balance_score=excluded.balance_score
			""",
			list(scores),
		)
		conn.commit()
