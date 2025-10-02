import sys
import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onair.config import load_config
from onair import db as dbmod


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""
	Calculate the great circle distance between two points on Earth in nautical miles.
	"""
	# Convert decimal degrees to radians
	lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
	
	# Haversine formula
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a))
	
	# Radius of earth in nautical miles
	r = 3440.065  # Earth's radius in nautical miles
	return c * r


def get_airport_coordinates(db_path: str, icao: str) -> Tuple[float, float]:
	"""Get latitude and longitude for an airport from the database."""
	with dbmod.connect(db_path) as conn:
		row = conn.execute("SELECT latitude, longitude FROM airports WHERE icao = ?", (icao,)).fetchone()
		if row and row[0] is not None and row[1] is not None:
			return float(row[0]), float(row[1])
		
		# Try to fetch from OnAir API if not found
		try:
			from onair.config import load_config
			from onair import api as api_mod
			
			cfg = load_config()
			api = api_mod.OnAirClient(cfg)
			airport_data = api.get_airport_by_icao(icao)
			
			if airport_data:
				# Cache the airport for future use
				dbmod.upsert_airport(db_path, airport_data)
				lat = airport_data.get('Latitude')
				lon = airport_data.get('Longitude')
				if lat is not None and lon is not None:
					return float(lat), float(lon)
		except Exception as e:
			pass  # Fall through to raise the original error
		
		raise ValueError(f"Airport {icao} not found in database and could not be fetched from API")


def get_fuel_weight_per_gallon(fuel_type: str) -> float:
	"""Get fuel weight per gallon based on fuel type."""
	fuel_weights = {
		'jet': 6.7,
		'100ll': 6.0,
		'avgas': 6.0,
		'jeta': 6.7,
		'jeta1': 6.7
	}
	return fuel_weights.get(fuel_type.lower(), 6.7)  # Default to jet fuel weight


class PerformanceOptimizer:
	"""Calculate performance curves for planes based on sample job data."""
	
	def __init__(self, db_path: str):
		self.db_path = db_path
		self.performance_curves = {}
		self.fuel_weights = {}
	
	def load_sample_jobs(self, excel_path: str) -> pd.DataFrame:
		"""Load sample jobs from Excel file."""
		df = pd.read_excel(excel_path, sheet_name='Jobs')
		return df
	
	def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
		"""Calculate performance metrics for each plane type."""
		performance_data = defaultdict(list)
		
		for _, row in df.iterrows():
			plane_type = row['Plane']
			departure = row['Departure']
			destination = row['Destination']
			time_hours = row['Time']
			gallons = row['Gallons']
			payload_lbs = row['Payload']
			dep_size = row['DepSize']
			dest_size = row['DestSize']
			
			try:
				# Calculate haversine distance
				lat1, lon1 = get_airport_coordinates(self.db_path, departure)
				lat2, lon2 = get_airport_coordinates(self.db_path, destination)
				haversine_dist = haversine_distance(lat1, lon1, lat2, lon2)
				
				# Get fuel type for this plane
				fuel_type = self._get_plane_fuel_type(plane_type)
				fuel_weight_per_gallon = get_fuel_weight_per_gallon(fuel_type)
				
				# Calculate total weight (payload + fuel)
				fuel_weight_lbs = gallons * fuel_weight_per_gallon
				total_weight = payload_lbs + fuel_weight_lbs
				
				# Calculate effective speed (distance / time)
				effective_speed = haversine_dist / time_hours
				
				# Store performance data
				performance_data[plane_type].append({
					'distance_nm': haversine_dist,
					'time_hours': time_hours,
					'effective_speed_kts': effective_speed,
					'payload_lbs': payload_lbs,
					'fuel_weight_lbs': fuel_weight_lbs,
					'total_weight_lbs': total_weight,
					'dep_size': dep_size,
					'dest_size': dest_size,
					'gallons': gallons
				})
				
			except Exception as e:
				print(f"Warning: Could not process job {departure}->{destination} for {plane_type}: {e}")
				continue
		
		return dict(performance_data)
	
	def _get_plane_fuel_type(self, plane_type: str) -> str:
		"""Get fuel type for a plane from the database."""
		with dbmod.connect(self.db_path) as conn:
			row = conn.execute("SELECT fuel_type FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
			if row and row[0]:
				return row[0]
			return 'jet'  # Default to jet fuel
	
	def calculate_performance_curves(self, performance_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
		"""Calculate performance curves for each plane type."""
		curves = {}
		
		for plane_type, jobs in performance_data.items():
			if len(jobs) < 2:  # Need at least 2 data points
				print(f"Warning: Not enough data points for {plane_type} ({len(jobs)} jobs)")
				continue
			
			# Extract data for analysis
			distances = [job['distance_nm'] for job in jobs]
			speeds = [job['effective_speed_kts'] for job in jobs]
			payloads = [job['payload_lbs'] for job in jobs]
			total_weights = [job['total_weight_lbs'] for job in jobs]
			fuel_weights = [job['fuel_weight_lbs'] for job in jobs]
			gallons = [job['gallons'] for job in jobs]
			dep_sizes = [job['dep_size'] for job in jobs]
			dest_sizes = [job['dest_size'] for job in jobs]
			
			# Calculate correlations and trends
			# Speed vs Distance correlation
			speed_distance_corr = np.corrcoef(distances, speeds)[0, 1] if len(distances) > 1 else 0
			
			# Speed vs Payload correlation
			speed_payload_corr = np.corrcoef(payloads, speeds)[0, 1] if len(payloads) > 1 else 0
			
			# Speed vs Total Weight correlation
			speed_weight_corr = np.corrcoef(total_weights, speeds)[0, 1] if len(total_weights) > 1 else 0
			
			# Airport size impact
			avg_dep_size = np.mean(dep_sizes)
			avg_dest_size = np.mean(dest_sizes)
			
			# Calculate base performance metrics
			avg_speed = np.mean(speeds)
			max_observed_speed = max(speeds)  # Maximum speed observed in sample data
			min_observed_speed = min(speeds)  # Minimum speed observed in sample data
			avg_distance = np.mean(distances)
			avg_payload = np.mean(payloads)
			avg_total_weight = np.mean(total_weights)
			avg_fuel_weight = np.mean(fuel_weights)
			avg_gallons = np.mean(gallons)
			
			# Calculate fuel burn rate (gallons per nautical mile)
			fuel_burn_rate_gal_per_nm = avg_gallons / avg_distance if avg_distance > 0 else 0
			
			# Calculate speed adjustment factors
			# These will be used to adjust the theoretical speed based on real-world factors
			speed_vs_distance_factor = self._calculate_speed_factor(distances, speeds)
			speed_vs_payload_factor = self._calculate_speed_factor(payloads, speeds)
			speed_vs_weight_factor = self._calculate_speed_factor(total_weights, speeds)
			
			# Calculate fuel consumption correlations
			fuel_vs_payload_factor = self._calculate_speed_factor(payloads, gallons)
			fuel_vs_distance_factor = self._calculate_speed_factor(distances, gallons)
			
			curves[plane_type] = {
				'avg_speed_kts': avg_speed,
				'max_observed_speed_kts': max_observed_speed,
				'min_observed_speed_kts': min_observed_speed,
				'avg_distance_nm': avg_distance,
				'avg_payload_lbs': avg_payload,
				'avg_total_weight_lbs': avg_total_weight,
				'avg_fuel_weight_lbs': avg_fuel_weight,
				'avg_gallons': avg_gallons,
				'avg_dep_size': avg_dep_size,
				'avg_dest_size': avg_dest_size,
				'fuel_burn_rate_gal_per_nm': fuel_burn_rate_gal_per_nm,
				'speed_distance_correlation': speed_distance_corr,
				'speed_payload_correlation': speed_payload_corr,
				'speed_weight_correlation': speed_weight_corr,
				'speed_vs_distance_factor': speed_vs_distance_factor,
				'speed_vs_payload_factor': speed_vs_payload_factor,
				'speed_vs_weight_factor': speed_vs_weight_factor,
				'fuel_vs_payload_factor': fuel_vs_payload_factor,
				'fuel_vs_distance_factor': fuel_vs_distance_factor,
				'sample_count': len(jobs),
				'jobs': jobs  # Store raw data for reference
			}
		
		return curves
	
	def _calculate_speed_factor(self, x_values: List[float], y_values: List[float]) -> Dict[str, float]:
		"""Calculate speed adjustment factors based on correlation."""
		if len(x_values) < 2:
			return {'slope': 0, 'intercept': np.mean(y_values) if y_values else 0}
		
		# Simple linear regression
		x_mean = np.mean(x_values)
		y_mean = np.mean(y_values)
		
		numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
		denominator = sum((x - x_mean) ** 2 for x in x_values)
		
		if denominator == 0:
			return {'slope': 0, 'intercept': y_mean}
		
		slope = numerator / denominator
		intercept = y_mean - slope * x_mean
		
		return {'slope': slope, 'intercept': intercept}
	
	def calculate_fuel_burn_for_leg(self, plane_type: str, distance_nm: float, payload_lbs: float) -> float:
		"""Calculate estimated fuel burn in gallons for a specific leg."""
		if plane_type not in self.performance_curves:
			return 0  # No data available
		
		curve = self.performance_curves[plane_type]
		
		# Use sample data to estimate fuel consumption rate (gallons per nautical mile)
		if 'fuel_burn_rate_gal_per_nm' in curve:
			base_fuel_rate = curve['fuel_burn_rate_gal_per_nm']
		else:
			# Fallback: calculate from average sample data
			if curve['avg_distance_nm'] > 0:
				base_fuel_rate = curve['avg_gallons'] / curve['avg_distance_nm']
			else:
				return 0
		
		# Adjust fuel burn based on payload (heavier = more fuel)
		if 'fuel_vs_payload_factor' in curve and curve['avg_payload_lbs'] > 0:
			payload_factor = curve['fuel_vs_payload_factor']
			payload_multiplier = 1 + (payload_factor['slope'] * (payload_lbs - curve['avg_payload_lbs']) / curve['avg_payload_lbs'])
			payload_multiplier = max(0.5, min(payload_multiplier, 2.0))  # Reasonable bounds
		else:
			payload_multiplier = 1.0
		
		# Calculate total fuel burn for this leg
		estimated_fuel_gallons = distance_nm * base_fuel_rate * payload_multiplier
		
		return max(0, estimated_fuel_gallons)  # Never negative
	
	def _get_default_max_speed(self, plane_type: str) -> float:
		"""Get default maximum cruise speed from plane specifications."""
		try:
			with dbmod.connect(self.db_path) as conn:
				row = conn.execute("SELECT speed_kts FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
				if row and row[0]:
					return float(row[0])
		except Exception:
			pass
		return 500  # Conservative fallback if no data available
	
	def get_optimized_speed(self, plane_type: str, distance_nm: float, payload_lbs: float, 
						   dep_size: int, dest_size: int) -> float:
		"""Get optimized speed calculation for a specific job, factoring in fuel weight."""
		if plane_type not in self.performance_curves:
			# Fall back to default max cruise speed from plane specs
			default_max = self._get_default_max_speed(plane_type)
			return min(default_max, 500)  # Conservative clamping for fallback
		
		curve = self.performance_curves[plane_type]
		
		# Calculate fuel requirements for this leg
		fuel_gallons = self.calculate_fuel_burn_for_leg(plane_type, distance_nm, payload_lbs)
		
		# Get fuel type and calculate fuel weight
		fuel_type = self._get_plane_fuel_type(plane_type)
		fuel_weight_per_gallon = get_fuel_weight_per_gallon(fuel_type)
		fuel_weight_lbs = fuel_gallons * fuel_weight_per_gallon
		
		# Calculate total weight (cargo + fuel)
		total_weight_lbs = payload_lbs + fuel_weight_lbs
		
		# Start with base speed
		base_speed = curve['avg_speed_kts']
		
		# Apply distance-based adjustment
		if 'speed_vs_distance_factor' in curve:
			distance_factor = curve['speed_vs_distance_factor']
			distance_adjustment = distance_factor['slope'] * (distance_nm - curve['avg_distance_nm'])
		else:
			distance_adjustment = 0
		
		# Apply total weight-based adjustment (this is the key enhancement)
		if 'speed_vs_weight_factor' in curve:
			weight_factor = curve['speed_vs_weight_factor']
			weight_adjustment = weight_factor['slope'] * (total_weight_lbs - curve['avg_total_weight_lbs'])
		else:
			weight_adjustment = 0
		
		# Apply airport size adjustment (smaller airports = slower operations)
		avg_airport_size = (dep_size + dest_size) / 2
		avg_curve_airport_size = (curve['avg_dep_size'] + curve['avg_dest_size']) / 2
		airport_size_adjustment = (avg_curve_airport_size - avg_airport_size) * 10  # 10 kts per size difference
		
		# Calculate final optimized speed
		optimized_speed = base_speed + distance_adjustment + weight_adjustment + airport_size_adjustment
		
		# Clamp to observed speed range from sample data
		max_speed = curve['max_observed_speed_kts']
		min_speed = curve['min_observed_speed_kts']
		
		# Apply clamping to observed range, with reasonable absolute bounds as fallback
		optimized_speed = max(min_speed, min(optimized_speed, max_speed))
		optimized_speed = max(50, min(optimized_speed, 2000))  # Absolute safety bounds
		
		return optimized_speed
	
	def load_and_process(self, excel_path: str) -> None:
		"""Load sample jobs and process performance curves."""
		print("Loading sample jobs from Excel...")
		df = self.load_sample_jobs(excel_path)
		
		print("Calculating performance metrics...")
		performance_data = self.calculate_performance_metrics(df)
		
		print("Calculating performance curves...")
		self.performance_curves = self.calculate_performance_curves(performance_data)
		
		print(f"Processed performance curves for {len(self.performance_curves)} plane types:")
		for plane_type, curve in self.performance_curves.items():
			print(f"  {plane_type}: {curve['sample_count']} samples, speed {curve['min_observed_speed_kts']:.0f}-{curve['max_observed_speed_kts']:.0f} kts (avg {curve['avg_speed_kts']:.1f}), fuel rate {curve['fuel_burn_rate_gal_per_nm']:.3f} gal/nm")


def main():
	"""Main function to test the performance optimizer."""
	config = load_config()
	optimizer = PerformanceOptimizer(config.db_path)
	
	# Load and process sample jobs
	excel_path = os.path.join(os.path.dirname(__file__), '..', 'planes.xlsx')
	optimizer.load_and_process(excel_path)
	
	# Test optimized speed calculation
	test_plane = 'Antonov AN-225-210'
	test_distance = 1000
	test_payload = 500000
	test_dep_size = 5
	test_dest_size = 4
	
	optimized_speed = optimizer.get_optimized_speed(test_plane, test_distance, test_payload, test_dep_size, test_dest_size)
	print(f"\nTest: {test_plane}")
	print(f"Distance: {test_distance} nm, Payload: {test_payload} lbs")
	print(f"Optimized speed: {optimized_speed:.1f} kts" if optimized_speed else "No optimization data available")


if __name__ == "__main__":
	main()


