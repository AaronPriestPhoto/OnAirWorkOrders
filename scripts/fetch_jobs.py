import argparse
import sys
from typing import Dict, Iterable, Set, Any

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
	finally:
		conn.close()


def _maybe_add_icao(val: Any, seen: Set[str]) -> None:
	if isinstance(val, str):
		code = val.strip().upper()
		if len(code) >= 2:
			seen.add(code)


def _recurse_for_icaos(obj: Any, seen: Set[str]) -> None:
	if isinstance(obj, dict):
		# Prefer explicit ICAO fields on objects
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
		# Common direct fields
		for key in ("Departure", "Destination", "departure", "destination"):
			val = job.get(key)
			if isinstance(val, str):
				_maybe_add_icao(val, seen)
		# Cargos legs often include airports with ICAO fields
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
		# Generic recursive search as fallback
		_recurse_for_icaos(job, seen)
	return seen


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
	client = OnAirClient(cfg)

	# Clear jobs table each online run for a fresh snapshot
	import sqlite3
	with sqlite3.connect(cfg.db_path) as conn:
		conn.execute("DELETE FROM jobs")
		conn.commit()

	inserted = 0
	icaos: Set[str] = set()

	if args.scope in ("company", "all"):
		jobs = client.list_company_jobs()
		inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source="company")
		icaos.update(_collect_icaos_from_jobs(jobs))

	if args.scope in ("fbos", "all"):
		fbos = client.list_fbos()
		for fbo in fbos:
			fbo_id = fbo.get("Id") or fbo.get("id") or fbo.get("FboId")
			if not fbo_id:
				continue
			jobs = client.list_fbo_jobs(str(fbo_id))
			inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source=f"fbo:{fbo_id}")
			icaos.update(_collect_icaos_from_jobs(jobs))

	cached = _ensure_airports_cached(cfg, client, icaos)

	print(f"Upserted {inserted} jobs into {cfg.db_path}")
	print(f"Airports referenced: {len(icaos)}; cached (existing+fetched): {cached}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
