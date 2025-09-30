from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import OnAirConfig


class OnAirApiError(Exception):
	pass


def _extract_list_payload(data: Any) -> Optional[List[Dict[str, Any]]]:
	# If it's already a list
	if isinstance(data, list):
		return data
	# Common wrappers observed: Content, Data, Items, results
	if isinstance(data, dict):
		for key in ("Content", "Data", "Items", "data", "items", "results", "Results"):
			val = data.get(key)
			if isinstance(val, list):
				return val
	return None


def _extract_object_payload(data: Any) -> Optional[Dict[str, Any]]:
	if isinstance(data, dict):
		for key in ("Content", "Data", "Item", "data", "item", "result", "Result"):
			val = data.get(key)
			if isinstance(val, dict):
				return val
		return data
	return None


class OnAirClient:
	def __init__(self, config: OnAirConfig):
		self._config = config
		self._session = requests.Session()
		self._session.headers.update({
			"oa-apikey": config.api_key,
			"Accept": "application/json",
			"User-Agent": "onair-python/0.1",
		})

	def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
		url = f"{self._config.base_url}{path}"
		retries = max(0, self._config.max_retries)
		last_exc: Optional[Exception] = None
		for attempt in range(retries + 1):
			try:
				resp = self._session.request(method, url, timeout=self._config.timeout_seconds, params=params)
				if resp.status_code >= 500:
					time.sleep(min(2 ** attempt, 5))
					continue
				return resp
			except requests.RequestException as exc:
				last_exc = exc
				time.sleep(min(2 ** attempt, 5))
		if last_exc:
			raise OnAirApiError(str(last_exc))
		raise OnAirApiError("Request failed without exception")

	def ping(self) -> Tuple[bool, Optional[int]]:
		try:
			resp = self._request("GET", "/ping")
			return True, resp.status_code
		except Exception:
			return False, None

	def get_company(self) -> Dict[str, Any]:
		resp = self._request("GET", f"/company/{self._config.company_id}")
		if resp.status_code != 200:
			raise OnAirApiError(f"Failed to get company: {resp.status_code} {resp.text}")
		return resp.json()

	def list_fbos(self) -> List[Dict[str, Any]]:
		paths = [
			f"/company/{self._config.company_id}/fbo",
			f"/company/{self._config.company_id}/fbos",
		]
		for p in paths:
			resp = self._request("GET", p)
			if resp.status_code == 200:
				data = resp.json()
				arr = _extract_list_payload(data)
				if arr is not None:
					return arr
			elif resp.status_code == 404:
				continue
			raise OnAirApiError(f"Unexpected response for {p}: {resp.status_code} {resp.text}")
		return []

	def list_company_jobs(self, *, page: Optional[int] = None) -> List[Dict[str, Any]]:
		params = {"page": page} if page is not None else None
		paths = [
			f"/company/{self._config.company_id}/jobs",
			f"/companies/{self._config.company_id}/jobs",
		]
		for p in paths:
			resp = self._request("GET", p, params=params)
			if resp.status_code == 200:
				data = resp.json()
				arr = _extract_list_payload(data)
				if arr is not None:
					return arr
			elif resp.status_code == 404:
				continue
			raise OnAirApiError(f"Unexpected response for {p}: {resp.status_code} {resp.text}")
		return []

	def list_fbo_jobs(self, fbo_id: str) -> List[Dict[str, Any]]:
		paths = [
			f"/fbo/{fbo_id}/jobs",
			f"/fbos/{fbo_id}/jobs",
			f"/company/{self._config.company_id}/fbo/{fbo_id}/jobs",
		]
		for p in paths:
			resp = self._request("GET", p)
			if resp.status_code == 200:
				data = resp.json()
				arr = _extract_list_payload(data)
				if arr is not None:
					return arr
			elif resp.status_code == 404:
				continue
			raise OnAirApiError(f"Unexpected response for {p}: {resp.status_code} {resp.text}")
		return []

	def get_airport_by_icao(self, icao: str) -> Dict[str, Any]:
		icao = (icao or "").strip().upper()
		paths = [
			f"/airport/{icao}",
			f"/airports/{icao}",
		]
		for p in paths:
			resp = self._request("GET", p)
			if resp.status_code == 200:
				data = _extract_object_payload(resp.json())
				if data is not None:
					return data
			elif resp.status_code == 404:
				continue
			raise OnAirApiError(f"Unexpected response for {p}: {resp.status_code} {resp.text}")
		raise OnAirApiError(f"Airport not found: {icao}")

	def list_company_fleet(self) -> List[Dict[str, Any]]:
		paths = [
			f"/company/{self._config.company_id}/fleet",
			f"/companies/{self._config.company_id}/fleet",
		]
		for p in paths:
			resp = self._request("GET", p)
			if resp.status_code == 200:
				data = resp.json()
				arr = _extract_list_payload(data) or (data if isinstance(data, list) else None)
				if arr is not None:
					return arr
			elif resp.status_code == 404:
				continue
			raise OnAirApiError(f"Unexpected response for {p}: {resp.status_code} {resp.text}")
		return []
