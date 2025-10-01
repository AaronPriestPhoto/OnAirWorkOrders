import argparse
import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onair.config import load_config
from onair import db as dbmod


COLUMN_MAP = {
	"Plane": "plane_type",
	"Speed": "cruise_speed_kts",
	"Airport": "min_airport_size",
	"Range1": "range1_nm",
	"Payload1": "payload1_lbs",
	"Range2": "range2_nm",
	"Payload2": "payload2_lbs",
	"Priority": "priority",
	"Fuel": "fuel_type",
}


def load_plane_specs_from_file(file_path: str = None, sheet_name: str = "Planes") -> int:
	"""Load plane specs from Excel file without argument parsing."""
	cfg = load_config()
	dbmod.init_db(cfg.db_path)
	dbmod.migrate_schema(cfg.db_path)
	
	# Get project root directory if no file path provided
	if file_path is None:
		project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		file_path = os.path.join(project_root, "planes.xlsx")

	if not os.path.exists(file_path):
		print(f"File not found: {file_path}")
		return 1

	df = pd.read_excel(file_path, sheet_name=sheet_name)
	df = df.rename(columns=COLUMN_MAP)
	for _, row in df.iterrows():
		plane_type = str(row.get("plane_type")).strip() if pd.notna(row.get("plane_type")) else None
		if not plane_type or plane_type.lower() in ("nan",):
			continue
		dbmod.upsert_plane_spec(
			cfg.db_path,
			plane_type=plane_type,
			cruise_speed_kts=(float(row.get("cruise_speed_kts")) if pd.notna(row.get("cruise_speed_kts")) else None),
			min_airport_size=(int(row.get("min_airport_size")) if pd.notna(row.get("min_airport_size")) else None),
			range1_nm=(float(row.get("range1_nm")) if pd.notna(row.get("range1_nm")) else None),
			payload1_lbs=(float(row.get("payload1_lbs")) if pd.notna(row.get("payload1_lbs")) else None),
			range2_nm=(float(row.get("range2_nm")) if pd.notna(row.get("range2_nm")) else None),
			payload2_lbs=(float(row.get("payload2_lbs")) if pd.notna(row.get("payload2_lbs")) else None),
			priority=(str(row.get("priority")).strip().lower() if pd.notna(row.get("priority")) else None),
			fuel_type=(str(row.get("fuel_type")).strip().lower() if pd.notna(row.get("fuel_type")) else None),
		)
	print("Plane specs loaded.")
	return 0


def main(argv=None) -> int:
	# Get project root directory
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	
	parser = argparse.ArgumentParser(description="Load plane specs from planes.xlsx Planes sheet")
	parser.add_argument("--file", default=os.path.join(project_root, "planes.xlsx"), help="Path to planes.xlsx")
	parser.add_argument("--sheet", default="Planes", help="Sheet name containing specs")
	args = parser.parse_args(argv)

	return load_plane_specs_from_file(args.file, args.sheet)


if __name__ == "__main__":
	sys.exit(main())
