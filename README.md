## OnAir Work Orders

Generate, score, and export OnAir Airline Manager company jobs (created with FBO queries) into optimized work orders with CSV/Excel outputs.
https://www.onair.company/

### Features
- Fetches jobs from OnAir API (company and FBOs) or uses cached data (offline)
- Caches airports, airplanes, and job legs into a local SQLite database (`onair_jobs.db`)
- Scores jobs for pay/XP efficiency and balances according to plane priorities
- Generates multi-job legs and sequential routes per plane
- Exports work orders to Excel (`.xlsx`) or CSV with readable, prioritized columns

### Requirements
- Python 3.10+
- An OnAir API key and Company ID

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
Copy `env.example` to `.env` and set your credentials:
```
ONAIR_API_KEY=your_api_key_here
ONAIR_COMPANY_ID=your_company_id_here
RUN_MODE=online  # or offline
DB_PATH=onair_jobs.db
AIRPORT_CACHE_DAYS=90
```
The app will read configuration via `onair/config.py`.

### Data Storage
- SQLite database: `onair_jobs.db`
  - Tables include: `jobs`, `job_legs`, `airports`, `airplanes`, `plane_specs`, `job_scores`
  - A backup may be created as `onair_jobs.db.backup` during certain operations

### Typical Workflow

Quick start:
- Simply run `python fetch_jobs.py`. By default it runs in ONLINE mode, fetches/caches data, scores jobs, and generates/exports work orders automatically. You only need to run individual scripts separately if you want finer control.

1) Fetch and cache data (ONLINE)
```bash
python fetch_jobs.py
```
This will:
- Clear previous data
- Fetch FBO jobs and fleet
- Cache airports referenced by jobs
- Build legs and distances
- Score jobs
- Optionally generate work orders automatically

2) Use cached data only (OFFLINE)
```bash
python fetch_jobs.py --mode offline
```
This will:
- Use existing `onair_jobs.db`
- Rebuild legs if needed
- Score jobs
- Optionally generate work orders

3) Generate work orders manually (optional; not needed for the quick start)
```bash
python work_order_generator.py --format excel --output workorders.xlsx
# or CSV
python work_order_generator.py --format csv --output workorders.csv
```
Useful flags:
- `--max-hours`: total hours per plane (default 24.0)
- `--epsilon-hours`: small overage allowance (default 0.5)
- `--no-export`: skip file creation

### Excel/CSV Export Details
Exports are created by `work_order_generator.py` with consistent headers, formatting, and row construction. The leading columns are readable and intentionally condensed:

Column order (left to right):
1. `plane_registration`
2. `work_order_id`
3. `job_sequence`
4. `job_type`
5. `source` (FBO ICAO when available)
6. `job_id` (first 5 characters only)
7. `departure`
8. `destination`
9. `route`
10. `legs_count`
11. `pay`
12. `xp`
13+. Remaining fields: `plane_type`, `priority`, `distance_nm`, `flight_hours`, `pay_per_hour`, `xp_per_hour`, `balance_score`, `time_remaining_hours`, `penalty_amount`, `adjusted_pay_per_hour`, `adjusted_pay_total`, `speed_kts`, `min_airport_size`, `accumulated_hours`, `is_late`, `payload_lbs`

Special handling:
- `plane_id` is not exported
- `source`: if the job `source` starts with `fbo:UUID`, it is replaced with the ICAO of the departure airport (for multi-leg rows: the leg’s `from_icao`)
- `job_id`: truncated to the first 5 characters
- Excel number formats are applied to currency and numeric columns

### Sample Data (Optional)
If present, `planes.xlsx` may include a `Jobs` sheet used by the performance optimizer to derive speed/fuel curves. This is optional; exports and work orders do not require it.

### Scripts
- `fetch_jobs.py`: Fetch/cache and optionally generate work orders
- `score_jobs.py`: Score jobs for efficiency per plane
- `work_order_generator.py`: Build optimized work orders and export to Excel/CSV
- `performance_optimizer.py`: Optional performance curves from sample data

### Troubleshooting
- If Excel/CSV files are open, exports will prompt to retry after closing the file.
- If airport cache is stale, airports will be refetched based on `AIRPORT_CACHE_DAYS`.

### License
This project is licensed under the MIT License. See `LICENSE` for details.


