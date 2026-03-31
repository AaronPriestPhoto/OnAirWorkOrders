from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_BASE_URL = "https://server1.onair.company/api/v1"


class OnAirApiError(RuntimeError):
	pass


def _maybe_load_dotenv() -> None:
	# Use existing .env if present (does nothing if python-dotenv isn't installed).
	try:
		from dotenv import load_dotenv  # type: ignore
	except Exception:
		return
	load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
	val = os.getenv(name)
	if val is None:
		return default
	val = val.strip()
	return val or default


def _request_json(base_url: str, api_key: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
	url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
	resp = requests.get(
		url,
		headers={"oa-apikey": api_key, "Accept": "application/json"},
		params=params,
		timeout=60,
	)
	if resp.status_code != 200:
		raise OnAirApiError(f"GET {url} failed: {resp.status_code} {resp.text[:500]}")
	try:
		return resp.json()
	except Exception as exc:  # pragma: no cover
		raise OnAirApiError(f"GET {url} returned non-JSON: {exc}") from exc


def _unwrap_content(payload: Any) -> Any:
	# OnAir commonly wraps responses in { Content: ... }
	if isinstance(payload, dict):
		for k in ("Content", "Data", "Items", "Results", "results", "items", "data"):
			if k in payload:
				return payload[k]
	return payload


def _list_aircraft_types(base_url: str, api_key: str) -> List[Dict[str, Any]]:
	"""
	Fetch aircraft types from OnAir.

	Public swagger only documents GET /aircrafttypes/{aircraftTypeId}, but in practice
	servers typically expose a list endpoint. We try a small set of common candidates.
	"""
	candidates = [
		"aircrafttypes",
		"aircraftTypes",
		"aircrafttypes/all",
		"aircraftTypes/all",
	]
	last_err: Optional[Exception] = None
	for path in candidates:
		try:
			data = _request_json(base_url, api_key, path)
			unwrapped = _unwrap_content(data)
			if isinstance(unwrapped, list):
				return [x for x in unwrapped if isinstance(x, dict)]
		except Exception as exc:
			last_err = exc
			continue

	raise OnAirApiError(
		"Unable to fetch aircraft type list from OnAir.\n"
		"Tried: "
		+ ", ".join(f"GET {DEFAULT_BASE_URL}/{p}" for p in candidates)
		+ "\n"
		+ (f"Last error: {last_err}" if last_err else "")
	)


def _addon_sim_short(addon: Dict[str, Any]) -> str:
	val = addon.get("SimVersionShortDisplay") or addon.get("SimVersionDisplay") or ""
	return str(val).strip().upper()


def _type_supports_sim(aircraft_type: Dict[str, Any], sim_short: str) -> bool:
	sim_short = (sim_short or "").strip().upper()
	if sim_short in ("", "ALL", "ALLSIMULATORS"):
		return True
	addons = aircraft_type.get("AircraftAddons") or []
	if not isinstance(addons, list):
		return False
	for addon in addons:
		if isinstance(addon, dict) and _addon_sim_short(addon) == sim_short:
			return True
	return False


def _safe_int(v: Any) -> Optional[int]:
	try:
		if v is None or v == "":
			return None
		return int(v)
	except Exception:
		return None


def _safe_float(v: Any) -> Optional[float]:
	try:
		if v is None or v == "":
			return None
		return float(v)
	except Exception:
		return None


def _row_for_aircraft_type(t: Dict[str, Any]) -> Dict[str, Any]:
	# Screenshot columns + min airport size
	display_name = t.get("DisplayName") or t.get("TypeName") or ""
	aircraft_class = t.get("AircraftClass") or ""
	return {
		"Type": str(display_name).strip(),
		"Class": str(aircraft_class).strip(),
		"Avg Price": _safe_float(t.get("Baseprice")),
		"Max Payload": _safe_float(t.get("maximumCargoWeight")),
		"Empty Weight": _safe_float(t.get("emptyWeight")),
		"Flights": _safe_int(t.get("FlightsCount")),
		"Range": _safe_float(t.get("maximumRangeInNM")),
		"Speed": _safe_float(t.get("designSpeedVC")),
		"Seats": _safe_int(t.get("seats")),
		"Copilot": bool(t.get("needsCopilot")) if t.get("needsCopilot") is not None else None,
		"Min Airport Size": _safe_int(t.get("AirportMinSize")),
	}


def export_aircraft_database_csv(
	*,
	base_url: str,
	api_key: str,
	output_path: str,
	simulator: str = "MSFS",
) -> int:
	types = _list_aircraft_types(base_url, api_key)
	types = [t for t in types if _type_supports_sim(t, simulator)]

	rows = [_row_for_aircraft_type(t) for t in types]
	rows = [r for r in rows if (r.get("Type") or "").strip()]
	rows.sort(key=lambda r: (r.get("Type") or "").lower())

	fieldnames = [
		"Type",
		"Class",
		"Avg Price",
		"Max Payload",
		"Empty Weight",
		"Flights",
		"Range",
		"Speed",
		"Seats",
		"Copilot",
		"Min Airport Size",
	]

	with open(output_path, "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
		w.writeheader()
		for r in rows:
			w.writerow(r)

	return len(rows)


def main(argv: Optional[Iterable[str]] = None) -> int:
	_maybe_load_dotenv()

	parser = argparse.ArgumentParser(
		description="Export OnAir aircraft market-style database to aircraft_database.csv (standalone, MSFS-only by default)."
	)
	parser.add_argument("--output", default="aircraft_database.csv", help="Output CSV path")
	parser.add_argument("--simulator", default="MSFS", help="All Simulators, MSFS, XP11, PSDv4+, FSX, PSDv3")
	parser.add_argument(
		"--base-url",
		default=_get_env("ONAIR_BASE_URL", DEFAULT_BASE_URL),
		help=f"OnAir API base URL (default: {DEFAULT_BASE_URL})",
	)
	parser.add_argument(
		"--api-key",
		default=_get_env("ONAIR_API_KEY"),
		help="OnAir API key (or set ONAIR_API_KEY env var)",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	if not args.api_key:
		raise SystemExit("Missing OnAir API key. Provide --api-key or set ONAIR_API_KEY.")

	count = export_aircraft_database_csv(
		base_url=args.base_url,
		api_key=args.api_key,
		output_path=args.output,
		simulator=args.simulator,
	)
	print(f"Wrote {count} aircraft types to {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

