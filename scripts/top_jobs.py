import argparse
import csv
import sqlite3
import json
from typing import Dict, List, Tuple

from onair.config import load_config


def _ensure_plane_specs_columns(conn: sqlite3.Connection) -> None:
	cols = {r[1] for r in conn.execute("PRAGMA table_info(plane_specs)").fetchall()}
	if "speed_kts" not in cols:
		conn.execute("ALTER TABLE plane_specs ADD COLUMN speed_kts REAL")
		conn.commit()


def _get_planes(conn: sqlite3.Connection, plane_type_filter: str | None) -> List[Dict]:
	rows = conn.execute("SELECT id, registration, type, model, data_json FROM airplanes").fetchall()
	planes: List[Dict] = []
	for r in rows:
		pid, reg, ptype, model, dj = r
		disp = None
		name = None
		try:
			data = json.loads(dj) if dj else {}
			disp = (data.get("AircraftType") or {}).get("DisplayName")
			name = (data.get("AircraftType") or {}).get("TypeName")
		except Exception:
			pass
		planes.append({"id": pid, "registration": reg, "type": ptype, "model": model, "display_name": disp, "type_name": name})
	if plane_type_filter:
		flt = plane_type_filter.strip()
		planes = [p for p in planes if (p.get("type") == flt or p.get("model") == flt or p.get("display_name") == flt or p.get("type_name") == flt)]
	return planes


def _get_priority(conn: sqlite3.Connection, plane_type: str) -> str:
	row = conn.execute("SELECT priority FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
	if row and row[0]:
		p = str(row[0]).strip().lower()
		if p in ("pay", "xp", "balance"):
			return p
	return "balance"


def _get_plane_speed_kts(conn: sqlite3.Connection, plane_id: str, plane_type: str | None) -> float:
	if plane_type:
		row = conn.execute("SELECT speed_kts FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
		if row and row[0]:
			try:
				return float(row[0])
			except Exception:
				pass
	row = conn.execute("SELECT data_json FROM airplanes WHERE id = ?", (plane_id,)).fetchone()
	if row and row[0]:
		import json as _json
		try:
			data = _json.loads(row[0])
			val = (
				data.get("AircraftType", {}).get("designSpeedVC")
				or data.get("AircraftType", {}).get("CruiseSpeedKts")
				or 0
			)
			return float(val or 0)
		except Exception:
			return 0.0
	return 0.0


def _fetch_scores_for_plane(conn: sqlite3.Connection, plane_id: str) -> List[Dict]:
	rows = conn.execute(
		"""
		SELECT js.job_id, js.plane_type, js.pay_per_hour, js.xp_per_hour, js.balance_score, j.source, IFNULL(j.computed_distance_nm,0), IFNULL(j.pay,0), IFNULL(j.xp,0)
		FROM job_scores js
		JOIN jobs j ON j.id = js.job_id
		WHERE js.plane_id = ? AND js.feasible = 1
		""",
		(plane_id,),
	).fetchall()
	return [
		{
			"job_id": r[0],
			"plane_type": r[1],
			"pay_per_hour": r[2] or 0.0,
			"xp_per_hour": r[3] or 0.0,
			"balance_score": r[4] or 0.0,
			"source": r[5],
			"distance_nm": r[6] or 0.0,
			"pay": r[7] or 0.0,
			"xp": r[8] or 0.0,
		}
		for r in rows
	]


def _route_for_job(conn: sqlite3.Connection, job_id: str) -> Tuple[str, int, float, int | None]:
	legs = conn.execute(
		"SELECT leg_index, from_icao, to_icao, IFNULL(distance_nm,0) FROM job_legs WHERE job_id = ? ORDER BY leg_index",
		(job_id,),
	).fetchall()
	if not legs:
		return ("", 0, 0.0, None)
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
	return (", ".join([c for c in route_seq if c]), len(legs), total_nm, min_ap_size)


def main() -> int:
	parser = argparse.ArgumentParser(description="Top jobs per plane")
	parser.add_argument("--plane-type", default=None, help="Filter to planes matching this type/model (DisplayName/TypeName supported)")
	parser.add_argument("--limit", type=int, default=5, help="Number of jobs to show per plane")
	parser.add_argument("--csv", default=None, help="Path to write CSV of results")
	cfg = load_config()
	conn = sqlite3.connect(cfg.db_path)
	_ensure_plane_specs_columns(conn)
	args = parser.parse_args()
	planes = _get_planes(conn, args.plane_type)
	if not planes:
		print("No planes found.")
		return 0

	csv_writer = None
	csv_file = None
	if args.csv:
		csv_file = open(args.csv, "w", newline="", encoding="utf-8")
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow([
			"plane_id", "plane_type", "registration", "job_id", "source", "distance_nm",
			"pay_per_hour", "xp_per_hour", "balance_score", "pay", "xp", "FlightHrs", "Speed", "MinAp", "route", "legs_count",
		])

	for plane in planes:
		scores = _fetch_scores_for_plane(conn, plane["id"]) 
		if not scores:
			continue
		ptype = args.plane_type or plane.get("type") or plane.get("model") or plane.get("display_name") or plane.get("type_name") or (scores[0]["plane_type"] if scores else None) or "unknown"
		priority = _get_priority(conn, ptype)
		if priority == "pay":
			key = lambda s: s["pay_per_hour"]
		elif priority == "xp":
			key = lambda s: s["xp_per_hour"]
		else:
			key = lambda s: s["balance_score"]
		# Ensure unique jobs by job_id
		sorted_scores = sorted(scores, key=key, reverse=True)
		seen_jobs = set()
		best = []
		for s in sorted_scores:
			jid = s["job_id"]
			if jid in seen_jobs:
				continue
			seen_jobs.add(jid)
			best.append(s)
			if len(best) >= max(1, args.limit):
				break

		print(f"Plane {plane['registration'] or plane['id']} [{ptype}] priority={priority}")
		speed_kts = _get_plane_speed_kts(conn, plane["id"], ptype)
		for i, s in enumerate(best, 1):
			route, legs_cnt, total_nm, min_ap = _route_for_job(conn, s["job_id"]) 
			flight_hrs = (total_nm / speed_kts) if speed_kts and speed_kts > 0 else 0.0
			min_ap_str = "" if min_ap is None else str(min_ap)
			print(
				f"  {i}. job={s['job_id']} source={s['source']} dist={s['distance_nm']:.0f}nm pay/hr={s['pay_per_hour']:.0f} xp/hr={s['xp_per_hour']:.0f} bal={s['balance_score']:.3f} hrs={flight_hrs:.2f} speed={speed_kts:.0f}kts minAp={min_ap_str} route={route}"
			)
			if csv_writer:
				csv_writer.writerow([
					plane["id"], ptype, plane.get("registration"), s["job_id"], s["source"], f"{s['distance_nm']:.2f}",
					f"{s['pay_per_hour']:.2f}", f"{s['xp_per_hour']:.2f}", f"{s['balance_score']:.6f}", f"{s['pay']:.2f}", f"{s['xp']:.2f}", f"{flight_hrs:.2f}", f"{speed_kts:.0f}", min_ap_str, route, legs_cnt,
				])
		print("")

	if csv_file:
		csv_file.close()
		print(f"CSV written: {args.csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
