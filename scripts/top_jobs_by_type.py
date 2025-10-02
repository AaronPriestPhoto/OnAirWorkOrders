import sys
import os
import argparse
import csv
import sqlite3
import json
from typing import Dict, List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onair.config import load_config
from performance_optimizer import PerformanceOptimizer


def _ensure_plane_specs_columns(conn: sqlite3.Connection) -> None:
	cols = {r[1] for r in conn.execute("PRAGMA table_info(plane_specs)").fetchall()}
	if "speed_kts" not in cols:
		conn.execute("ALTER TABLE plane_specs ADD COLUMN speed_kts REAL")
		conn.commit()


def _priority_for_type(conn: sqlite3.Connection, plane_type: str) -> str:
	row = conn.execute("SELECT priority FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
	if row and row[0]:
		p = str(row[0]).strip().lower()
		if p in ("pay", "xp", "balance"):
			return p
	return "balance"


def _speed_for_type_or_plane(conn: sqlite3.Connection, plane_type: str | None, plane_id: str | None) -> float:
	if plane_type:
		row = conn.execute("SELECT speed_kts, cruise_speed_kts FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
		if row:
			s1, s2 = row[0], row[1]
			for v in (s1, s2):
				if v:
					try:
						return float(v)
					except Exception:
						pass
	if plane_id:
		row = conn.execute("SELECT data_json FROM airplanes WHERE id = ?", (plane_id,)).fetchone()
		if row and row[0]:
			try:
				data = json.loads(row[0])
				val = (
					data.get("AircraftType", {}).get("designSpeedVC")
					or data.get("AircraftType", {}).get("CruiseSpeedKts")
					or 0
				)
				return float(val or 0)
			except Exception:
				return 0.0
	return 0.0


def _route_for_job(conn: sqlite3.Connection, job_id: str) -> Tuple[str, int, float, int | None, str, str]:
	legs = conn.execute(
		"SELECT leg_index, from_icao, to_icao, IFNULL(distance_nm,0) FROM job_legs WHERE job_id = ? ORDER BY leg_index",
		(job_id,),
	).fetchall()
	if not legs:
		return ("", 0, 0.0, None, "", "")
	points: List[str] = []
	total_nm = 0.0
	min_ap_size: int | None = None
	for idx, f, t, dnm in legs:
		from_code = (f or '').upper()
		to_code = (t or '').upper()
		if not points:
			points.append(from_code)
		points.append(to_code)
		total_nm += float(dnm or 0.0)
		for icao in (from_code, to_code):
			if not icao:
				continue
			row = conn.execute("SELECT size FROM airports WHERE icao = ?", (icao,)).fetchone()
			if row is not None and row[0] is not None:
				sz = int(row[0])
				min_ap_size = sz if (min_ap_size is None or sz < min_ap_size) else min_ap_size
	route_seq: List[str] = []
	for code in points:
		if not route_seq or route_seq[-1] != code:
			route_seq.append(code)
	departure = route_seq[0] if route_seq else ""
	destination = route_seq[-1] if route_seq else ""
	return (", ".join([c for c in route_seq if c]), len(legs), total_nm, min_ap_size, departure, destination)


def _plane_type_label(js_plane_type: str | None, airplane_data_json: str | None) -> str:
	if js_plane_type and str(js_plane_type).strip():
		return str(js_plane_type).strip()
	try:
		data = json.loads(airplane_data_json) if airplane_data_json else {}
		at = data.get("AircraftType") or {}
		return at.get("DisplayName") or at.get("TypeName") or "unknown"
	except Exception:
		return "unknown"


def main() -> int:
	parser = argparse.ArgumentParser(description="Export top 10 jobs for every plane type")
	parser.add_argument("--csv", default="top10_by_type.csv", help="Path to write CSV of results (default: top10_by_type.csv)")
	args = parser.parse_args()

	cfg = load_config()
	conn = sqlite3.connect(cfg.db_path)
	_ensure_plane_specs_columns(conn)
	
	# Initialize performance optimizer
	optimizer = PerformanceOptimizer(cfg.db_path)
	excel_path = os.path.join(os.path.dirname(__file__), '..', 'planes.xlsx')
	optimizer.load_and_process(excel_path)

	# Load all feasible scores joined with jobs and airplanes
	rows = conn.execute(
		"""
		SELECT js.plane_id, js.plane_type, js.job_id, js.pay_per_hour, js.xp_per_hour, js.balance_score,
		       j.source, IFNULL(j.computed_distance_nm,0), IFNULL(j.pay,0), IFNULL(j.xp,0), a.data_json
		FROM job_scores js
		JOIN jobs j ON j.id = js.job_id
		LEFT JOIN airplanes a ON a.id = js.plane_id
		WHERE js.feasible = 1
		"""
	).fetchall()

	# Group scores by resolved plane type label
	by_type: Dict[str, List[Dict]] = {}
	for r in rows:
		plane_id, js_type, job_id, pph, xph, bal, src, dist_nm, pay, xp, adata = r
		ptype = _plane_type_label(js_type, adata)
		by_type.setdefault(ptype, []).append({
			"plane_id": plane_id,
			"plane_type": ptype,
			"job_id": job_id,
			"pay_per_hour": pph or 0.0,
			"xp_per_hour": xph or 0.0,
			"balance_score": bal or 0.0,
			"source": src,
			"distance_nm": dist_nm or 0.0,
			"pay": pay or 0.0,
			"xp": xp or 0.0,
		})

	# Prepare CSV
	with open(args.csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"plane_type", "job_id", "source", "distance_nm",
			"pay_per_hour", "xp_per_hour", "balance_score", "pay", "xp",
			"FlightHrs", "Speed", "MinAp", "Departure", "Destination", "route", "legs_count",
		])

		for ptype, scores in sorted(by_type.items(), key=lambda kv: kv[0]):
			priority = _priority_for_type(conn, ptype)
			if priority == "pay":
				key = lambda s: s["pay_per_hour"]
			elif priority == "xp":
				key = lambda s: s["xp_per_hour"]
			else:
				key = lambda s: s["balance_score"]
			# Enforce unique jobs by job_id per plane type
			sorted_scores = sorted(scores, key=key, reverse=True)
			seen_jobs = set()
			top = []
			for s in sorted_scores:
				jid = s["job_id"]
				if jid in seen_jobs:
					continue
				seen_jobs.add(jid)
				top.append(s)
				if len(top) >= 10:
					break

			for s in top:
				route, legs_cnt, total_nm, min_ap, departure, destination = _route_for_job(conn, s["job_id"]) 
				
				# Calculate optimized flight hours per leg if optimizer is available
				base_speed_kts = _speed_for_type_or_plane(conn, ptype, s["plane_id"]) or 0.0
				if optimizer and ptype in optimizer.performance_curves:
					# Get detailed leg information for per-leg optimization
					legs = conn.execute("""
						SELECT distance_nm, cargo_lbs, from_icao, to_icao 
						FROM job_legs 
						WHERE job_id = ? 
						ORDER BY leg_index
					""", (s["job_id"],)).fetchall()
					
					total_flight_hrs = 0.0
					total_optimized_distance = 0.0
					
					for leg in legs:
						leg_distance = leg[0] or 0
						leg_payload = leg[1] or 0
						
						# Get airport sizes for this leg (use min_ap as fallback)
						dep_size = min_ap or 0
						dest_size = min_ap or 0
						
						# Calculate optimized speed for this specific leg
						leg_speed = optimizer.get_optimized_speed(ptype, leg_distance, leg_payload, dep_size, dest_size)
						if leg_speed and leg_speed > 0:
							leg_flight_hrs = leg_distance / leg_speed
							total_flight_hrs += leg_flight_hrs
							total_optimized_distance += leg_distance
					
					# Calculate average speed for display
					flight_hrs = total_flight_hrs
					speed_kts = total_optimized_distance / total_flight_hrs if total_flight_hrs > 0 else base_speed_kts
				else:
					# Fallback to old calculation
					speed_kts = base_speed_kts
					flight_hrs = (total_nm / speed_kts) if speed_kts and speed_kts > 0 else 0.0
				writer.writerow([
					ptype, s["job_id"], s["source"], f"{s['distance_nm']:.2f}",
					f"{s['pay_per_hour']:.2f}", f"{s['xp_per_hour']:.2f}", f"{s['balance_score']:.6f}", f"{s['pay']:.2f}", f"{s['xp']:.2f}",
					f"{flight_hrs:.2f}", f"{speed_kts:.0f}", ("" if min_ap is None else min_ap), departure, destination, route, legs_cnt,
				])

	print(f"CSV written: {args.csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
