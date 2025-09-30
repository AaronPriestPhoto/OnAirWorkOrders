import argparse
import math
import sys
from typing import Dict, Iterable, Set, Any, List, Tuple

from onair.config import load_config
from onair.api import OnAirClient, OnAirApiError
from onair import db as dbmod


def _print_db_stats(db_path: str) -> None:
	import sqlite3
	conn = sqlite3.connect(db_path)
	try:
		c = conn.cursor()
		c.execute("select count(*) from jobs")
		total = c.fetchone()[0]
		print(f"Total jobs: {total}")
		print("By source:")
		for src, cnt in c.execute("select source, count(*) from jobs group by source order by count(*) desc"):
			print(f"  {src}: {cnt}")
		c.execute("select count(*) from airplanes")
		print(f"Airplanes: {c.fetchone()[0]}")
		try:
			c.execute("select count(*) from job_legs")
			print(f"Job legs: {c.fetchone()[0]}")
		except Exception:
			pass
	finally:
		conn.close()


def _maybe_add_icao(val: Any, seen: Set[str]) -> None:
	if isinstance(val, str):
		code = val.strip().upper()
		if len(code) >= 2:
			seen.add(code)


def _recurse_for_icaos(obj: Any, seen: Set[str]) -> None:
	if isinstance(obj, dict):
		for k in ("ICAO", "icao", "Icao"):
			if k in obj and isinstance(obj[k], str):
				_maybe_add_icao(obj[k], seen)
		for v in obj.values():
			_recurse_for_icaos(v, seen)
	elif isinstance(obj, list):
		for it in obj:
			_recurse_for_icaos(it, seen)


def _collect_icaos_from_jobs(jobs: Iterable[Dict]) -> Set[str]:
	seen: Set[str] = set()
	for job in jobs:
		for key in ("Departure", "Destination", "departure", "destination"):
			val = job.get(key)
			if isinstance(val, str):
				_maybe_add_icao(val, seen)
		for cargo_key in ("Cargos", "cargos"):
			cargos = job.get(cargo_key) or []
			if isinstance(cargos, list):
				for cg in cargos:
					for ap_key in ("CurrentAirport", "DepartureAirport", "DestinationAirport", "Airport"):
						ap = isinstance(cg, dict) and cg.get(ap_key) or None
						if isinstance(ap, dict):
							icao = ap.get("ICAO") or ap.get("icao")
							if isinstance(icao, str):
								_maybe_add_icao(icao, seen)
		_recurse_for_icaos(job, seen)
	return seen


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	# Earth radius nautical miles
	R = 3440.065
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlam = math.radians(lon2 - lon1)
	a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return R * c


def _extract_job_legs(job: Dict[str, Any]) -> List[Tuple[str, str]]:
	legs: List[Tuple[str, str]] = []
	# Prefer cargos sequence if present
	cargos = job.get("Cargos") or job.get("cargos")
	if isinstance(cargos, list) and len(cargos) > 0:
		# Group cargos by sequence of airports if available
		seq: List[str] = []
		for cg in cargos:
			dep = None
			dst = None
			dep_ap = isinstance(cg, dict) and (cg.get("DepartureAirport") or cg.get("CurrentAirport"))
			dst_ap = isinstance(cg, dict) and cg.get("DestinationAirport")
			if isinstance(dep_ap, dict):
				dep = dep_ap.get("ICAO") or dep_ap.get("icao")
			if isinstance(dst_ap, dict):
				dst = dst_ap.get("ICAO") or dst_ap.get("icao")
			if isinstance(dep, str):
				seq.append(dep.strip().upper())
			if isinstance(dst, str):
				seq.append(dst.strip().upper())
		# Collapse into legs
		for i in range(len(seq) - 1):
			legs.append((seq[i], seq[i+1]))
	# Fallback to overall departure/destination
	dep = job.get("Departure") or job.get("departure")
	dst = job.get("Destination") or job.get("destination")
	if isinstance(dep, str) and isinstance(dst, str):
		legs.append((dep.strip().upper(), dst.strip().upper()))
	return legs


def _compute_and_store_leg_distances(cfg, jobs: Iterable[Dict[str, Any]]) -> int:
	count = 0
	# Clear legs before recompute
	dbmod.clear_job_legs(cfg.db_path)
	for job in jobs:
		job_id = job.get("Id") or job.get("id") or job.get("JobId")
		if not job_id:
			continue
		pairs = _extract_job_legs(job)
		total_nm = 0.0
		leg_index = 0
		for a, b in pairs:
			if not a or not b:
				continue
			a_ap = dbmod.get_airport(cfg.db_path, a)
			b_ap = dbmod.get_airport(cfg.db_path, b)
			if not a_ap or not b_ap or a_ap.get("latitude") is None or a_ap.get("longitude") is None or b_ap.get("latitude") is None or b_ap.get("longitude") is None:
				continue
			d_nm = _haversine_nm(float(a_ap["latitude"]), float(a_ap["longitude"]), float(b_ap["latitude"]), float(b_ap["longitude"]))
			dbmod.upsert_job_leg(cfg.db_path, str(job_id), leg_index, a, b, float(a_ap["latitude"]), float(a_ap["longitude"]), float(b_ap["latitude"]), float(b_ap["longitude"]), float(d_nm))
			total_nm += d_nm
			leg_index += 1
		if leg_index > 0:
			dbmod.update_job_total_distance(cfg.db_path, str(job_id), float(total_nm))
			count += leg_index
	return count


def _ensure_airports_cached(cfg, client: OnAirClient, icaos: Set[str]) -> int:
	cached_or_fetched = 0
	for icao in sorted(icaos):
		if not icao:
			continue
		if not dbmod.is_airport_stale(cfg.db_path, icao, cfg.airport_cache_days):
			cached_or_fetched += 1
			continue
		ap = client.get_airport_by_icao(icao)
		if isinstance(ap, dict) and "Airport" in ap and isinstance(ap["Airport"], dict):
			ap = ap["Airport"]
		dbmod.upsert_airport(cfg.db_path, ap)
		cached_or_fetched += 1
	return cached_or_fetched


def main(argv=None) -> int:
	cfg = load_config()
	parser = argparse.ArgumentParser(description="Fetch OnAir jobs into SQLite")
	parser.add_argument("--scope", choices=["company", "fbos", "all"], default="all", help="Which job scope to fetch")
	parser.add_argument("--mode", choices=["online", "offline"], default=cfg.run_mode, help="Run mode: online hits API, offline uses cached DB only")
	args = parser.parse_args(argv)

	if args.mode == "offline":
		_print_db_stats(cfg.db_path)
		return 0

	dbmod.init_db(cfg.db_path)
	dbmod.migrate_schema(cfg.db_path)
	client = OnAirClient(cfg)

	# Clear jobs and airplanes tables each online run for a fresh snapshot
	import sqlite3
	with sqlite3.connect(cfg.db_path) as conn:
		conn.execute("DELETE FROM jobs")
		conn.execute("DELETE FROM airplanes")
		conn.commit()

	inserted = 0
	icaos: Set[str] = set()
	all_jobs: List[Dict[str, Any]] = []

	if args.scope in ("company", "all"):
		jobs = client.list_company_jobs()
		all_jobs.extend(jobs)
		inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source="company")
		icaos.update(_collect_icaos_from_jobs(jobs))

	if args.scope in ("fbos", "all"):
		fbos = client.list_fbos()
		for fbo in fbos:
			fbo_id = fbo.get("Id") or fbo.get("id") or fbo.get("FboId")
			if not fbo_id:
				continue
			jobs = client.list_fbo_jobs(str(fbo_id))
			all_jobs.extend(jobs)
			inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source=f"fbo:{fbo_id}")
			icaos.update(_collect_icaos_from_jobs(jobs))

	cached = _ensure_airports_cached(cfg, client, icaos)

	# Fetch fleet
	fleet = client.list_company_fleet()
	planes = dbmod.upsert_airplanes(cfg.db_path, fleet)

	# Compute leg distances and per-job totals
	legs_created = _compute_and_store_leg_distances(cfg, all_jobs)

	print(f"Upserted {inserted} jobs into {cfg.db_path}")
	print(f"Airports referenced: {len(icaos)}; cached (existing+fetched): {cached}")
	print(f"Airplanes upserted: {planes}")
	print(f"Job legs stored: {legs_created}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
