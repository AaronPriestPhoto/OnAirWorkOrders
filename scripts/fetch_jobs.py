import argparse
import sys

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

	inserted = 0
	if args.scope in ("company", "all"):
		jobs = client.list_company_jobs()
		inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source="company")

	if args.scope in ("fbos", "all"):
		fbos = client.list_fbos()
		for fbo in fbos:
			fbo_id = fbo.get("Id") or fbo.get("id") or fbo.get("FboId")
			if not fbo_id:
				continue
			jobs = client.list_fbo_jobs(str(fbo_id))
			inserted += dbmod.upsert_jobs(cfg.db_path, jobs, source=f"fbo:{fbo_id}")

	print(f"Upserted {inserted} jobs into {cfg.db_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
