import argparse
import sys

from onair.config import load_config
from onair.api import OnAirClient, OnAirApiError
from onair import db as dbmod


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description="Fetch OnAir jobs into SQLite")
	parser.add_argument("--scope", choices=["company", "fbos", "all"], default="all", help="Which job scope to fetch")
	args = parser.parse_args(argv)

	cfg = load_config()
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
