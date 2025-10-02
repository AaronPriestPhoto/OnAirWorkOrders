import sys
import os
import json
from collections import defaultdict
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onair import config as cfg_mod
from onair import db as db_mod
from performance_optimizer import PerformanceOptimizer


def _load_airports(conn):
	rows = conn.execute('SELECT icao, size FROM airports').fetchall()
	return {r[0]: (r[1] if r[1] is not None else 0) for r in rows}


def _load_jobs_and_legs(conn):
	# jobs
	jobs = {}
	for r in conn.execute('SELECT id, pay, xp, computed_distance_nm FROM jobs').fetchall():
		jobs[r[0]] = {"pay": r[1] or 0.0, "xp": r[2] or 0.0, "computed_distance_nm": r[3] or 0.0}
	# legs grouped
	legs_by_job = defaultdict(list)
	for r in conn.execute('SELECT job_id, leg_index, from_icao, to_icao, distance_nm, cargo_lbs FROM job_legs ORDER BY job_id, leg_index').fetchall():
		legs_by_job[r[0]].append({
			"from_icao": r[2],
			"to_icao": r[3],
			"distance_nm": r[4] or 0.0,
			"cargo_lbs": r[5] or 0.0,
		})
	return jobs, legs_by_job


def _load_planes(conn):
	planes = []
	for r in conn.execute('SELECT id, type, data_json FROM airplanes').fetchall():
		planes.append({"id": r[0], "type": r[1], "data_json": r[2]})
	return planes


def _get_plane_specs(conn, plane_type_display_name, plane_type_name, api_plane_data):
	# If no connection provided, only build from API plane data
	if conn is None:
		if api_plane_data:
			return {
				"speed_kts": api_plane_data.get("AircraftType", {}).get("designSpeedVC", 0) or 0,
				"min_airport_size": api_plane_data.get("AircraftType", {}).get("AirportMinSize", 0) or 0,
				"range1_nm": api_plane_data.get("AircraftType", {}).get("maximumRangeInNM", 0) or 0,
				"payload1_lbs": api_plane_data.get("AircraftType", {}).get("maximumCargoWeight", 0) or 0,
				"range2_nm": api_plane_data.get("AircraftType", {}).get("maximumRangeInNM", 0) or 0,
				"payload2_lbs": api_plane_data.get("AircraftType", {}).get("maximumCargoWeight", 0) or 0,
				"priority": "balance"
			}
		return None
	cursor = conn.cursor()
	spec = cursor.execute('SELECT * FROM plane_specs WHERE plane_type = ?', (plane_type_display_name,)).fetchone()
	if not spec:
		spec = cursor.execute('SELECT * FROM plane_specs WHERE plane_type = ?', (plane_type_name,)).fetchone()
	if spec:
		columns = [d[0] for d in cursor.description] if cursor.description else []
		def col(name, alt=None):
			try:
				idx = columns.index(name)
				return spec[idx]
			except Exception:
				if alt:
					try:
						idx = columns.index(alt)
						return spec[idx]
					except Exception:
						return None
				return None
		speed = col("cruise_speed_kts", alt="speed_kts") or 0
		return {
			"speed_kts": speed,
			"min_airport_size": col("min_airport_size") or 0,
			"range1_nm": col("range1_nm") or 0,
			"payload1_lbs": col("payload1_lbs") or 0,
			"range2_nm": col("range2_nm") or 0,
			"payload2_lbs": col("payload2_lbs") or 0,
			"priority": col("priority") or "balance",
		}
	if api_plane_data:
		return {
			"speed_kts": api_plane_data.get("AircraftType", {}).get("designSpeedVC", 0) or 0,
			"min_airport_size": api_plane_data.get("AircraftType", {}).get("AirportMinSize", 0) or 0,
			"range1_nm": api_plane_data.get("AircraftType", {}).get("maximumRangeInNM", 0) or 0,
			"payload1_lbs": api_plane_data.get("AircraftType", {}).get("maximumCargoWeight", 0) or 0,
			"range2_nm": api_plane_data.get("AircraftType", {}).get("maximumRangeInNM", 0) or 0,
			"payload2_lbs": api_plane_data.get("AircraftType", {}).get("maximumCargoWeight", 0) or 0,
			"priority": "balance"
		}
	return None


def _calculate_payload_range_limit(plane_spec, distance_nm):
	# Get the maximum range and maximum payload for bounds checking
	max_range = max(plane_spec["range1_nm"] or 0, plane_spec["range2_nm"] or 0)
	max_payload = max(plane_spec["payload1_lbs"] or 0, plane_spec["payload2_lbs"] or 0)
	
	# If distance exceeds maximum range, no payload possible
	if distance_nm > max_range:
		return 0
	
	# If we have identical ranges, return max payload if within range
	if plane_spec["range1_nm"] == plane_spec["range2_nm"]:
		return max_payload
	
	# If we have identical payloads, return that payload if within range
	if plane_spec["payload1_lbs"] == plane_spec["payload2_lbs"]:
		return plane_spec["payload1_lbs"]
	
	# Sort the range/payload pairs to ensure r1 <= r2
	if plane_spec["range1_nm"] > plane_spec["range2_nm"]:
		r1, p1 = plane_spec["range2_nm"], plane_spec["payload2_lbs"]
		r2, p2 = plane_spec["range1_nm"], plane_spec["payload1_lbs"]
	else:
		r1, p1 = plane_spec["range1_nm"], plane_spec["payload1_lbs"]
		r2, p2 = plane_spec["range2_nm"], plane_spec["payload2_lbs"]
	
	# Handle edge cases
	if r1 == r2:  # Same range, different payloads
		return max_payload if distance_nm <= r1 else 0
	
	# Linear interpolation between the two points
	if distance_nm <= r1:
		return max(p1, p2)  # Use maximum payload for short distances
	elif distance_nm >= r2:
		return min(p1, p2)  # Use minimum payload for long distances
	else:
		# Linear interpolation: payload decreases as distance increases
		m = (p2 - p1) / (r2 - r1)
		calculated_payload = p1 + m * (distance_nm - r1)
		# Ensure we never exceed the maximum payload
		return min(calculated_payload, max_payload)


def _score_job_for_plane(airports, plane_spec, job_id, job, legs, optimizer=None, plane_type=None):
	feasible = 1
	reason = "OK"
	base_speed = plane_spec["speed_kts"] or 0
	min_sz = int(plane_spec.get("min_airport_size") or 0)
	# Use the maximum of both ranges to determine if job is feasible
	max_range = max(plane_spec["range1_nm"] or 0, plane_spec["range2_nm"] or 0)
	if (job["computed_distance_nm"] or 0) > max_range:
		return (0, "Job distance exceeds max range", 0.0, 0.0, 0.0)
	flight_hours = 0.0
	for leg in legs:
		f = (leg["from_icao"] or "").upper()
		t = (leg["to_icao"] or "").upper()
		# airport sizes
		fsz = airports.get(f, 0)
		tsz = airports.get(t, 0)
		if fsz < min_sz:
			return (0, f"Departure {f} below min airport size {min_sz}", 0.0, 0.0, 0.0)
		if tsz < min_sz:
			return (0, f"Destination {t} below min airport size {min_sz}", 0.0, 0.0, 0.0)
		# payload
		cap = _calculate_payload_range_limit(plane_spec, leg["distance_nm"] or 0)
		if (leg["cargo_lbs"] or 0) > cap:
			return (0, f"Leg cargo exceeds capacity for {leg['distance_nm']:.0f}nm", 0.0, 0.0, 0.0)
		
		# Use optimized speed if available, otherwise fall back to base speed
		if optimizer and plane_type:
			optimized_speed = optimizer.get_optimized_speed(
				plane_type, 
				leg["distance_nm"] or 0, 
				leg["cargo_lbs"] or 0, 
				fsz, 
				tsz
			)
			speed = optimized_speed if optimized_speed else base_speed
		else:
			speed = base_speed
		
		if speed <= 0:
			return (0, "Plane speed is 0", 0.0, 0.0, 0.0)
		flight_hours += (leg["distance_nm"] or 0) / speed
	if flight_hours <= 0:
		return (0, "Computed flight time is 0", 0.0, 0.0, 0.0)
	pph = (job["pay"] or 0.0) / flight_hours
	xph = (job["xp"] or 0.0) / flight_hours
	bal = (pph / 1_000_000 + xph / 100) / 2 if (pph > 0 and xph > 0) else (pph or xph)
	return (1, "OK", pph, xph, bal)


def main():
	config = cfg_mod.load_config()
	db_path = config.db_path
	
	# Initialize performance optimizer
	optimizer = PerformanceOptimizer(db_path)
	excel_path = os.path.join(os.path.dirname(__file__), '..', 'planes.xlsx')
	if os.path.exists(excel_path):
		print("Loading performance optimization data...")
		optimizer.load_and_process(excel_path)
		print(f"Loaded optimization data for {len(optimizer.performance_curves)} plane types")
	else:
		print("No performance optimization data found, using default calculations")
	
	with db_mod.connect(db_path) as conn:
		airports = _load_airports(conn)
		jobs, legs_by_job = _load_jobs_and_legs(conn)
		planes = _load_planes(conn)
		# Preload plane specs we need into a map
		plane_specs_cache = {}
		cur = conn.cursor()
		for p in set([pl["type"] for pl in planes if pl.get("type")]):
			sp = _get_plane_specs(conn, p, p, {})
			if sp:
				plane_specs_cache[p] = sp

	# Compute in memory with progress bar
	bulk = []
	total_combinations = len(planes) * len(jobs)
	scored_count = 0
	
	# Create a progress bar for the scoring process
	progress = tqdm(total=total_combinations, desc="Scoring jobs", unit="combinations")
	
	for pl in planes:
		ptype = pl.get("type")
		api_data = json.loads(pl["data_json"]) if pl.get("data_json") else {}
		spec = plane_specs_cache.get(ptype) or _get_plane_specs(None, None, None, api_data)  # fallback from API data
		if not spec:
			# Still need to update progress for skipped planes
			progress.update(len(jobs))
			continue
			
		for job_id, job in jobs.items():
			legs = legs_by_job.get(job_id, [])
			if not legs:
				progress.update(1)
				continue
			feas, reason, pph, xph, bal = _score_job_for_plane(airports, spec, job_id, job, legs, optimizer, ptype)
			bulk.append((pl["id"], ptype, job_id, int(feas), reason, pph, xph, bal))
			scored_count += 1
			progress.update(1)
			progress.set_postfix({"Plane": (ptype or "Unknown")[:20], "Scored": scored_count})
	
	progress.close()

	# Write in bulk
	db_mod.clear_job_scores(db_path)
	if bulk:
		db_mod.upsert_job_scores_bulk(db_path, bulk)
	print(f"Scores computed for {len(planes)} planes (all statuses)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
