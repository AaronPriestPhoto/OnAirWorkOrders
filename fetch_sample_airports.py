import sys
import os
import pandas as pd
from tqdm import tqdm

# Scripts are now in the root directory - no path modification needed

from onair.config import load_config
from onair import api as api_mod
from onair import db as db_mod


def get_missing_airports_from_sample_jobs(excel_path: str) -> set:
    """Extract all unique airports from sample jobs that aren't in the database."""
    # Read sample jobs
    df = pd.read_excel(excel_path, sheet_name='Jobs')
    
    # Get all unique airports from departure and destination columns
    airports_in_jobs = set()
    for _, row in df.iterrows():
        dep = str(row.get('Departure', '')).strip().upper()
        dest = str(row.get('Destination', '')).strip().upper()
        if dep and dep != 'NAN':
            airports_in_jobs.add(dep)
        if dest and dest != 'NAN':
            airports_in_jobs.add(dest)
    
    # Check which ones are missing from database
    cfg = load_config()
    missing_airports = set()
    
    with db_mod.connect(cfg.db_path) as conn:
        for airport in airports_in_jobs:
            row = conn.execute("SELECT icao FROM airports WHERE icao = ?", (airport,)).fetchone()
            if not row:
                missing_airports.add(airport)
    
    return missing_airports


def fetch_and_cache_airports(missing_airports: set) -> int:
    """Fetch missing airports from OnAir API and cache them."""
    if not missing_airports:
        print("No missing airports to fetch.")
        return 0
    
    cfg = load_config()
    api = api_mod.OnAirClient(cfg)
    
    cached_count = 0
    failed_airports = []
    
    print(f"Fetching {len(missing_airports)} missing airports from OnAir API...")
    
    for airport_icao in tqdm(missing_airports, desc="Fetching airports"):
        try:
            # Try to get airport info from OnAir API
            airport_data = api.get_airport_by_icao(airport_icao)
            
            if airport_data:
                # Cache the airport
                db_mod.upsert_airport(cfg.db_path, airport_data)
                cached_count += 1
                print(f"  + Cached {airport_icao}: {airport_data.get('Name', 'Unknown')}")
            else:
                failed_airports.append(airport_icao)
                print(f"  - Failed to fetch {airport_icao}")
                
        except Exception as e:
            failed_airports.append(airport_icao)
            print(f"  - Error fetching {airport_icao}: {e}")
    
    if failed_airports:
        print(f"\nFailed to fetch {len(failed_airports)} airports: {', '.join(failed_airports)}")
        print("These airports may not exist in OnAir or may have different ICAO codes.")
    
    return cached_count


def main():
    """Main function to fetch missing airports from sample jobs."""
    # Get current directory (where planes.xlsx is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "planes.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found")
        return 1
    
    print("Analyzing sample jobs for missing airports...")
    missing_airports = get_missing_airports_from_sample_jobs(excel_path)
    
    if not missing_airports:
        print("All airports from sample jobs are already cached!")
        return 0
    
    print(f"Found {len(missing_airports)} missing airports: {', '.join(sorted(missing_airports))}")
    
    # Ask user for confirmation
    response = input("\nFetch these airports from OnAir API? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0
    
    # Fetch and cache the missing airports
    cached_count = fetch_and_cache_airports(missing_airports)
    
    print(f"\nSuccessfully cached {cached_count} airports!")
    print("You can now re-run the performance optimizer to use all sample job data.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
