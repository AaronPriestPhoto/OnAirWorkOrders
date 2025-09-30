import os
from dataclasses import dataclass
from typing import Optional

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	# If python-dotenv is not installed, skip silently; env can be provided by OS
	pass


@dataclass(frozen=True)
class OnAirConfig:
	api_key: str
	company_id: str
	base_url_override: Optional[str]
	timeout_seconds: int
	max_retries: int
	db_path: str
	run_mode: str
	airport_cache_days: int

	@property
	def base_url(self) -> str:
		if self.base_url_override:
			return self.base_url_override.rstrip("/")
		# Default to server1 if no override is provided
		return "https://server1.onair.company/api/v1"


_def_timeout = int(os.getenv("ONAIR_TIMEOUT_SECONDS", "30"))
_def_retries = int(os.getenv("ONAIR_MAX_RETRIES", "3"))
_def_airport_cache_days = int(os.getenv("ONAIR_AIRPORT_CACHE_DAYS", "90"))


def load_config() -> OnAirConfig:
	api_key = os.getenv("ONAIR_API_KEY", "").strip()
	company_id = os.getenv("ONAIR_COMPANY_ID", "").strip()
	base_url_override = os.getenv("ONAIR_BASE_URL")
	db_path = os.getenv("ONAIR_DB_PATH", "onair_jobs.db").strip()
	run_mode = os.getenv("ONAIR_RUN_MODE", "online").strip().lower()
	if run_mode not in ("online", "offline"):
		run_mode = "online"

	if not api_key and run_mode == "online":
		raise ValueError("ONAIR_API_KEY is required (set it in .env or environment)")
	if not company_id and run_mode == "online":
		raise ValueError("ONAIR_COMPANY_ID is required (set it in .env or environment)")

	return OnAirConfig(
		api_key=api_key,
		company_id=company_id,
		base_url_override=base_url_override,
		timeout_seconds=_def_timeout,
		max_retries=_def_retries,
		db_path=db_path,
		run_mode=run_mode,
		airport_cache_days=_def_airport_cache_days,
	)
