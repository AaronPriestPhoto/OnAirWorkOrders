import sys
import os
import argparse
import json
import math
from datetime import datetime, timedelta
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onair import config as cfg_mod
from onair import api as api_mod
from onair import db as db_mod


def _print_offline_stats(db_path: str) -> None:
	with db_mod.connect(db_path) as conn:
		c = conn.cursor()
		def _count(table: str) -> int:
			row = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
			return int(row[0]) if row else 0
		jobs = _count("jobs")
		airports = _count("airports")
		airplanes = _count("airplanes")
		job_legs = _count("job_legs")
		print(f"Total jobs: {jobs}")
		if jobs > 0:
			for row in c.execute('SELECT source, COUNT(*) FROM jobs GROUP BY source ORDER BY COUNT(*) DESC'):
				print(f"  {row[0]}: {row[1]}")
		print(f"Airports cached: {airports}")
		print(f"Airplanes cached: {airplanes}")
		print(f"Job legs stored: {job_legs}")


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	R = 3440.065
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlam = math.radians(lon2 - lon1)
	a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return R * c


def _extract_legs_from_job(job: dict) -> list[dict]:
	legs: list[dict] = []
	job_id = job.get('Id') or job.get('id') or job.get('JobId')
	cargo_weight = job.get('TotalCargoWeight') or job.get('CargoWeight') or job.get('Cargo') or 0.0
	if job.get('Cargos'):
		for idx, cargo in enumerate(job['Cargos']):
			f = (cargo.get('DepartureAirport') or {}).get('ICAO')
			t = (cargo.get('DestinationAirport') or {}).get('ICAO')
			w = cargo.get('Weight') or 0.0
			if f and t:
				legs.append({"job_id": job_id, "leg_index": idx, "from_icao": f, "to_icao": t, "cargo_lbs": w})
	elif job.get('Charters'):
		for idx, ch in enumerate(job['Charters']):
			f = (ch.get('DepartureAirport') or {}).get('ICAO')
			t = (ch.get('DestinationAirport') or {}).get('ICAO')
			if f and t:
				legs.append({"job_id": job_id, "leg_index": idx, "from_icao": f, "to_icao": t, "cargo_lbs": 0.0})
	else:
		f = job.get('Departure') or job.get('departure')
		t = job.get('Destination') or job.get('destination')
		if f and t:
			legs.append({"job_id": job_id, "leg_index": 0, "from_icao": f, "to_icao": t, "cargo_lbs": cargo_weight})
	return legs


def _build_job_legs_and_distances(db_path: str) -> int:
	"""Parse stored job JSON, build legs, compute distances, and update totals."""
	# Clear existing legs
	db_mod.clear_job_legs(db_path)
	inserted = 0
	with db_mod.connect(db_path) as conn:
		cur = conn.cursor()
		rows = cur.execute("SELECT id, data_json FROM jobs").fetchall()
		for row in rows:
			job_id = row[0]
			try:
				job = json.loads(row[1])
			except Exception:
				continue
			legs = _extract_legs_from_job(job)
			total_nm = 0.0
			for leg in legs:
				f = leg["from_icao"].strip().upper()
				t = leg["to_icao"].strip().upper()
				ap_from = cur.execute("SELECT latitude, longitude FROM airports WHERE icao = ?", (f,)).fetchone()
				ap_to = cur.execute("SELECT latitude, longitude FROM airports WHERE icao = ?", (t,)).fetchone()
				from_lat = ap_from[0] if ap_from else None
				from_lon = ap_from[1] if ap_from else None
				to_lat = ap_to[0] if ap_to else None
				to_lon = ap_to[1] if ap_to else None
				dist = 0.0
				if from_lat is not None and from_lon is not None and to_lat is not None and to_lon is not None:
					dist = _haversine_nm(from_lat, from_lon, to_lat, to_lon)
				db_mod.upsert_job_leg(db_path, leg["job_id"], leg["leg_index"], f, t, from_lat or 0.0, from_lon or 0.0, to_lat or 0.0, to_lon or 0.0, dist, leg.get("cargo_lbs"))
				total_nm += dist
			inserted += len(legs)
			db_mod.update_job_total_distance(db_path, job_id, total_nm)
	return inserted


def _is_cargo_only_job(job: dict) -> bool:
	"""Check if a job is cargo-only (no passenger/PAX requirements)."""
	# Check for passenger jobs
	total_pax = job.get("TotalPaxTransported", 0)
	charters = job.get("Charters", [])
	
	if total_pax > 0 or len(charters) > 0:
		return False
	
	# If we get here, it's cargo-only
	return True


def _is_automated_job(job: dict) -> bool:
	"""Check if a job is automated (no human-only requirements)."""
	# Check for human-only jobs (check both job level and cargo level)
	if job.get("HumanOnly", False):
		return False
	
	# Check for human-only cargo items
	cargos = job.get("Cargos", [])
	for cargo in cargos:
		if cargo.get("HumanOnly", False):
			return False
	
	# If we get here, it's automated
	return True


def main():
	parser = argparse.ArgumentParser(description="Fetch jobs from OnAir API and store in SQLite.")
	parser.add_argument("--mode", type=str, choices=["online", "offline"],
					help="Run mode: 'online' to fetch from API, 'offline' to use cached data.")
	parser.add_argument("--enable-passenger-jobs", action="store_true", 
					help="Include passenger/PAX jobs (default: cargo-only)")
	parser.add_argument("--enable-human-only-jobs", action="store_true", 
					help="Include human-only jobs (default: automated only)")
	args = parser.parse_args()

	config = cfg_mod.load_config()
	db_mod.init_db(config.db_path)
	db_mod.migrate_schema(config.db_path)
	client = api_mod.OnAirClient(config)

	run_mode = args.mode if args.mode else config.run_mode

	if run_mode == 'offline':
		print("Running in OFFLINE mode. Using cached data only.")
		_print_offline_stats(config.db_path)
		
		# Load plane specs (always needed for scoring)
		print("Loading plane specs...")
		from scripts.load_plane_specs import load_plane_specs_from_file
		load_plane_specs_from_file()
		
		# Try to build legs from existing data too
		print("Building job legs...")
		built = _build_job_legs_and_distances(config.db_path)
		print(f"Rebuilt job legs from cache: {built}")
		# Verify all ICAOs for legs have airport rows
		with db_mod.connect(config.db_path) as conn:
			missing = _verify_airports_for_jobs(conn)
			if missing:
				print(f"WARNING: {len(missing)} ICAOs referenced in jobs are missing in airports (offline): {', '.join(sorted(missing))}")
		return 0

	print("Running in ONLINE mode. Fetching fresh data from API.")
	
	# Clear tables for fresh data in online mode
	with db_mod.connect(config.db_path) as conn:
		conn.execute('DELETE FROM jobs')
		conn.execute('DELETE FROM job_legs')
		conn.execute('DELETE FROM airplanes')
		conn.commit()

	all_jobs = []

	# Fetch FBO jobs
	print("Fetching FBOs...")
	fbos = client.list_fbos()
	print(f"Found {len(fbos)} FBOs. Fetching jobs for each...")
	
	# Add progress bar for FBO job fetching
	fbo_progress = tqdm(fbos, desc="Fetching FBO jobs", unit="FBO", position=1, leave=False)
	total_fbo_jobs = 0
	for fbo in fbo_progress:
		fbo_id = fbo['Id']
		icao = fbo['Airport']['ICAO']
		fbo_jobs = client.list_fbo_jobs(fbo_id)
		
		# Apply filtering (both enabled by default, can be disabled)
		original_count = len(fbo_jobs)
		filtered_jobs = fbo_jobs
		
		# Apply cargo-only filtering (default: enabled)
		if not args.enable_passenger_jobs:
			filtered_jobs = [job for job in filtered_jobs if _is_cargo_only_job(job)]
		
		# Apply human-only filtering (default: enabled)
		if not args.enable_human_only_jobs:
			filtered_jobs = [job for job in filtered_jobs if _is_automated_job(job)]
		
		fbo_jobs = filtered_jobs
		filtered_count = original_count - len(fbo_jobs)
		
		if filtered_count > 0:
			filter_types = []
			if not args.enable_passenger_jobs:
				filter_types.append("passenger")
			if not args.enable_human_only_jobs:
				filter_types.append("human-only")
			filter_desc = " and ".join(filter_types)
			tqdm.write(f"  {icao}: filtered out {filtered_count} {filter_desc} jobs")
		
		all_jobs.extend(fbo_jobs)
		inserted = db_mod.upsert_jobs(config.db_path, fbo_jobs, source=f"fbo:{fbo_id}")
		total_fbo_jobs += inserted
		fbo_progress.set_postfix({"ICAO": icao, "Jobs": inserted, "Total": total_fbo_jobs})
		# Print detailed info above the progress bar
		tqdm.write(f"  {icao}: {inserted} jobs")
	
	fbo_progress.close()
	print(f"Total FBO jobs upserted: {total_fbo_jobs}")

	# Fetch airplanes
	print("Fetching company fleet...")
	airplanes = client.list_company_fleet()
	inserted_airplanes = db_mod.upsert_airplanes(config.db_path, airplanes)
	print(f"Airplanes upserted: {inserted_airplanes}")

	# Load plane specs
	print("Loading plane specs...")
	from scripts.load_plane_specs import load_plane_specs_from_file
	load_plane_specs_from_file()

	# Extract all unique ICAOs from fetched jobs (including nested legs)
	unique_icaos = set()
	for job in all_jobs:
		if job.get('Departure'):
			unique_icaos.add(job['Departure'])
		if job.get('Destination'):
			unique_icaos.add(job['Destination'])
		if job.get('Cargos'):
			for cargo in job['Cargos']:
				if (cargo.get('DepartureAirport') or {}).get('ICAO'):
					unique_icaos.add(cargo['DepartureAirport']['ICAO'])
				if (cargo.get('DestinationAirport') or {}).get('ICAO'):
					unique_icaos.add(cargo['DestinationAirport']['ICAO'])
		if job.get('Charters'):
			for ch in job['Charters']:
				if (ch.get('DepartureAirport') or {}).get('ICAO'):
					unique_icaos.add(ch['DepartureAirport']['ICAO'])
				if (ch.get('DestinationAirport') or {}).get('ICAO'):
					unique_icaos.add(ch['DestinationAirport']['ICAO'])

	print(f"Found {len(unique_icaos)} unique ICAOs in jobs. Checking cache...")

	# Fetch and cache airport data
	print(f"Processing {len(unique_icaos)} unique ICAOs for airport data...")
	with db_mod.connect(config.db_path) as conn:
		cursor = conn.cursor()
		cached_airports_count = 0
		fetched_airports_count = 0

		# Add progress bar for airport fetching
		airport_progress = tqdm(unique_icaos, desc="Processing airports", unit="ICAO", position=0)
		for icao in airport_progress:
			airport_data = db_mod.get_airport(config.db_path, icao)
			is_stale = db_mod.is_airport_stale(config.db_path, icao, config.airport_cache_days) if airport_data else True
			if airport_data and not is_stale:
				cached_airports_count += 1
				airport_progress.set_postfix({"ICAO": icao, "Status": "cached", "Fetched": fetched_airports_count, "Cached": cached_airports_count})
				# Print detailed info above the progress bar
				from datetime import datetime, timezone, timedelta
				if airport_data and airport_data.get("updated_at"):
					try:
						updated = datetime.fromisoformat(airport_data["updated_at"]).replace(tzinfo=timezone.utc)
						age_days = (datetime.now(timezone.utc) - updated).days
						tqdm.write(f"  {icao}: using cached data (age: {age_days} days)")
					except Exception:
						tqdm.write(f"  {icao}: using cached data")
				else:
					tqdm.write(f"  {icao}: using cached data")
			else:
				try:
					api_airport_data = client.get_airport_by_icao(icao)
					db_mod.upsert_airport(config.db_path, api_airport_data)
					fetched_airports_count += 1
					airport_progress.set_postfix({"ICAO": icao, "Status": "fetched", "Fetched": fetched_airports_count, "Cached": cached_airports_count})
					# Print detailed info above the progress bar
					if airport_data and is_stale:
						tqdm.write(f"  {icao}: fetched new data (cache expired >{config.airport_cache_days} days)")
					else:
						tqdm.write(f"  {icao}: fetched new data (not previously cached)")
				except api_mod.OnAirApiError as e:
					airport_progress.set_postfix({"ICAO": icao, "Status": "error", "Fetched": fetched_airports_count, "Cached": cached_airports_count})
					tqdm.write(f"Warning: Could not fetch airport data for {icao}: {e}")

		# Build legs and distances now that airports are cached
		print("Building job legs...")
		built = _build_job_legs_and_distances(config.db_path)
		print(f"Job legs built: {built}")

		# Post-fetch verification: ensure every ICAO exists in airports
		missing_after_fetch = _verify_airports_for_jobs(conn)
		if missing_after_fetch:
			print(f"WARNING: {len(missing_after_fetch)} ICAOs referenced in jobs are missing in airports after fetch: {', '.join(sorted(missing_after_fetch))}")
		else:
			print("All referenced ICAOs are present in airports table.")

	print(f"Airports referenced: {len(unique_icaos)}; cached (existing+fetched): {cached_airports_count + fetched_airports_count}")
	with db_mod.connect(config.db_path) as conn:
		print(f"Job legs stored: {conn.execute('SELECT COUNT(*) FROM job_legs').fetchone()[0]}")

	return 0


def _verify_airports_for_jobs(conn):
	"""Return a set of ICAOs referenced by job legs that are missing in airports."""
	missing = set()
	for row in conn.execute('SELECT DISTINCT from_icao, to_icao FROM job_legs'):
		for icao in (row[0], row[1]):
			if not icao:
				continue
			ar = conn.execute('SELECT 1 FROM airports WHERE icao = ?', (icao,)).fetchone()
			if not ar:
				missing.add(icao)
	return missing


if __name__ == "__main__":
	sys.exit(main())
