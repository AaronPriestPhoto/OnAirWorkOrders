#!/usr/bin/env python3
"""
Work Order Generator for OnAir Airline Manager

This script generates optimized work orders for idle planes by chaining together
the most profitable jobs based on each plane's priority (Pay, XP, or Balanced).

Features:
- Chains jobs sequentially from destination to departure
- Respects configurable time limits (default 24 hours with 0.5hr epsilon)
- Prioritizes planes by efficiency: Pay > XP > Balanced, then by tail number
- Ensures no job reuse across work orders
- Sorts jobs within work orders by time remaining to minimize penalties
- Supports two-pass optimization for better time utilization
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from math import sqrt

from onair.config import load_config
from onair import db as db_mod
from performance_optimizer import PerformanceOptimizer


@dataclass
class JobInfo:
    """Information about a job for work order planning."""
    job_id: str
    departure: str
    destination: str
    pay: float
    xp: float
    distance_nm: float
    pay_per_hour: float
    xp_per_hour: float
    balance_score: float
    flight_hours: float
    time_remaining_hours: float
    penalty_amount: float = 0.0
    adjusted_pay_per_hour: float = 0.0
    adjusted_pay_total: float = 0.0
    source: str = ""
    speed_kts: float = 0.0
    min_airport_size: Optional[int] = None
    route: str = ""
    legs_count: int = 0
    hub_penalty_hours: float = 0.0
    legs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlaneInfo:
    """Information about a plane for work order planning."""
    plane_id: str
    registration: str
    plane_type: str
    priority: str  # 'pay', 'xp', or 'balance'
    current_location: str
    status: int
    feasible_jobs: List[JobInfo] = field(default_factory=list)


@dataclass
class WorkOrder:
    """A work order containing a sequence of chained jobs for a plane."""
    plane_id: str
    plane_registration: str
    plane_type: str
    priority: str
    jobs: List[JobInfo] = field(default_factory=list)
    total_hours: float = 0.0
    total_pay: float = 0.0
    total_xp: float = 0.0
    total_adjusted_pay: float = 0.0
    
    def add_job(self, job: JobInfo, accumulated_hours: float = 0.0) -> None:
        """Add a job to the work order and update totals."""
        # Calculate penalty based on accumulated time
        job.penalty_amount = self._calculate_penalty(job, accumulated_hours + job.flight_hours)
        job.adjusted_pay_total = job.pay - job.penalty_amount
        job.adjusted_pay_per_hour = job.adjusted_pay_total / job.flight_hours if job.flight_hours > 0 else 0.0
        
        self.jobs.append(job)
        self.total_hours += job.flight_hours
        self.total_pay += job.pay
        self.total_xp += job.xp
        self.total_adjusted_pay += job.adjusted_pay_total
    
    def _calculate_penalty(self, job: JobInfo, hours_from_now: float, penalty_hours_to_max: float = 24.0) -> float:
        """Calculate penalty for a job based on when it will be completed."""
        hours_late = max(0.0, hours_from_now - job.time_remaining_hours)
        if hours_late > 0:
            penalty_percentage = min(1.0, hours_late / penalty_hours_to_max)
            # We need to get the max penalty from the job data - for now use a reasonable estimate
            max_penalty = job.pay * 0.5  # Assume max penalty is 50% of pay
            return max_penalty * penalty_percentage
        return 0.0
    
    def get_current_location(self) -> str:
        """Get the current location after all jobs in the work order."""
        return self.jobs[-1].destination if self.jobs else ""
    
    def can_add_job(self, job: JobInfo, max_hours: float, epsilon_hours: float = 0.5) -> bool:
        """Check if a job can be added within time constraints."""
        new_total = self.total_hours + job.flight_hours
        return new_total <= (max_hours + epsilon_hours)
    
    def get_priority_score(self, job: JobInfo) -> float:
        """Get the priority score for a job based on plane priority."""
        if self.priority == "pay":
            return job.adjusted_pay_per_hour if hasattr(job, 'adjusted_pay_per_hour') else job.pay_per_hour
        elif self.priority == "xp":
            return job.xp_per_hour
        else:  # balance
            return job.balance_score


class WorkOrderGenerator:
    """Generates optimized work orders for idle planes."""
    
    def __init__(self, db_path: str, optimizer: Optional[PerformanceOptimizer] = None):
        self.db_path = db_path
        self.optimizer = optimizer
        self.used_jobs: Set[str] = set()
        
    def get_idle_planes(self) -> List[PlaneInfo]:
        """Get all idle planes sorted by priority and registration."""
        with db_mod.connect(self.db_path) as conn:
            # Get planes with status 0 (available/idle)
            rows = conn.execute("""
                SELECT id, registration, type, data_json 
                FROM airplanes 
                ORDER BY registration
            """).fetchall()
            
            planes = []
            for plane_id, registration, plane_type, data_json in rows:
                try:
                    data = json.loads(data_json) if data_json else {}
                    aircraft_status = data.get('AircraftStatus', 0)
                    
                    # Only include available/idle planes (status 0)
                    if aircraft_status != 0:
                        continue
                    
                    # Get current location
                    current_location = data.get('CurrentAirportICAO', '')
                    if not current_location:
                        # Try to find nearest airport using coordinates
                        lat = data.get('Latitude')
                        lon = data.get('Longitude')
                        if lat is not None and lon is not None:
                            current_location = self._find_nearest_airport(conn, float(lat), float(lon))
                        
                        if not current_location:
                            continue  # Skip planes without location
                    
                    # Get plane priority from specs
                    priority = self._get_plane_priority(conn, plane_type)
                    
                    plane = PlaneInfo(
                        plane_id=plane_id,
                        registration=registration or plane_id,
                        plane_type=plane_type or 'Unknown',
                        priority=priority,
                        current_location=current_location,
                        status=aircraft_status
                    )
                    
                    planes.append(plane)
                    
                except Exception as e:
                    print(f"Warning: Error processing plane {plane_id}: {e}")
                    continue
            
            # Sort by priority (Pay > XP > Balanced) then by registration
            priority_order = {'pay': 0, 'xp': 1, 'balance': 2}
            planes.sort(key=lambda p: (priority_order.get(p.priority, 3), p.registration))
            
            return planes
    
    def _get_plane_priority(self, conn: sqlite3.Connection, plane_type: str) -> str:
        """Get the priority setting for a plane type."""
        row = conn.execute("SELECT priority FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
        if row and row[0]:
            priority = str(row[0]).strip().lower()
            if priority in ("pay", "xp", "balance"):
                return priority
        return "balance"  # Default
    
    def _find_nearest_airport(self, conn: sqlite3.Connection, lat: float, lon: float) -> Optional[str]:
        """Find the nearest airport to given coordinates."""
        # Get all airports with coordinates
        airports = conn.execute("""
            SELECT icao, latitude, longitude 
            FROM airports 
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """).fetchall()
        
        if not airports:
            return None
        
        min_distance = float('inf')
        nearest_icao = None
        
        for icao, airport_lat, airport_lon in airports:
            # Simple distance calculation (good enough for finding nearest)
            distance = sqrt((lat - airport_lat)**2 + (lon - airport_lon)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_icao = icao
        
        return nearest_icao
    
    def _calculate_distance_between_airports(self, icao1: str, icao2: str) -> float:
        """Calculate distance between two airports using Haversine formula."""
        with db_mod.connect(self.db_path) as conn:
            # Get coordinates for both airports
            coords1 = conn.execute("SELECT latitude, longitude FROM airports WHERE icao = ?", (icao1,)).fetchone()
            coords2 = conn.execute("SELECT latitude, longitude FROM airports WHERE icao = ?", (icao2,)).fetchone()
            
            if not coords1 or not coords2:
                return 0.0
            
            lat1, lon1 = coords1
            lat2, lon2 = coords2
            
            # Haversine formula for great circle distance
            from math import radians, cos, sin, asin, sqrt
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Earth's radius in nautical miles
            r = 3440.065  # nm
            distance = c * r
            
            return distance
    
    def load_feasible_jobs_for_plane(self, plane: PlaneInfo) -> None:
        """Load all feasible jobs for a specific plane."""
        with db_mod.connect(self.db_path) as conn:
            # Get feasible jobs for this plane
            rows = conn.execute("""
                SELECT js.job_id, js.pay_per_hour, js.xp_per_hour, js.balance_score,
                       j.departure, j.destination, j.pay, j.xp, j.computed_distance_nm, j.source
                FROM job_scores js
                JOIN jobs j ON j.id = js.job_id
                WHERE js.plane_id = ? AND js.feasible = 1
                ORDER BY js.balance_score DESC
            """, (plane.plane_id,)).fetchall()
            
            jobs = []
            for row in rows:
                job_id, pph, xph, balance, dep, dest, pay, xp, dist, source = row
                
                # Skip jobs that are already used
                if job_id in self.used_jobs:
                    continue
                
                # Get job legs
                legs = self._get_job_legs(conn, job_id)
                
                # Calculate flight hours and speed
                flight_hours, avg_speed = self._calculate_flight_hours_and_speed(plane, legs)
                
                # Calculate time remaining
                time_remaining = self._calculate_time_remaining_hours(conn, job_id)
                
                # Get actual departure and destination from legs
                actual_departure = legs[0]['from_icao'] if legs else (dep or '')
                actual_destination = legs[-1]['to_icao'] if legs else (dest or '')
                
                # Calculate route and min airport size
                route_str, min_airport_size = self._calculate_route_info(conn, legs)
                
                job_info = JobInfo(
                    job_id=job_id,
                    departure=actual_departure,
                    destination=actual_destination,
                    pay=pay or 0.0,
                    xp=xp or 0.0,
                    distance_nm=dist or 0.0,
                    pay_per_hour=pph or 0.0,
                    xp_per_hour=xph or 0.0,
                    balance_score=balance or 0.0,
                    flight_hours=flight_hours,
                    time_remaining_hours=time_remaining,
                    source=source or '',
                    speed_kts=avg_speed,
                    min_airport_size=min_airport_size,
                    route=route_str,
                    legs_count=len(legs),
                    legs=legs
                )
                
                jobs.append(job_info)
            
            plane.feasible_jobs = jobs
    
    def _get_job_legs(self, conn: sqlite3.Connection, job_id: str) -> List[Dict[str, Any]]:
        """Get legs for a job."""
        rows = conn.execute("""
            SELECT leg_index, from_icao, to_icao, distance_nm, cargo_lbs
            FROM job_legs 
            WHERE job_id = ? 
            ORDER BY leg_index
        """, (job_id,)).fetchall()
        
        return [
            {
                'leg_index': row[0],
                'from_icao': row[1] or '',
                'to_icao': row[2] or '',
                'distance_nm': row[3] or 0.0,
                'cargo_lbs': row[4] or 0.0
            }
            for row in rows
        ]
    
    def _calculate_flight_hours_and_speed(self, plane: PlaneInfo, legs: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate total flight hours and average speed for job legs."""
        if not legs:
            return 0.0, 0.0
        
        total_hours = 0.0
        total_distance = 0.0
        
        for leg in legs:
            distance = leg['distance_nm']
            payload = leg['cargo_lbs']
            total_distance += distance
            
            # Get optimized speed if available
            if self.optimizer and plane.plane_type:
                # For airport sizes, we'll use a default or try to look them up
                dep_size = 3  # Default medium airport
                dest_size = 3  # Default medium airport
                
                speed = self.optimizer.get_optimized_speed(
                    plane.plane_type, distance, payload, dep_size, dest_size
                )
                
                # If optimizer returns 0 (plane not in performance curves), fall back to base speed
                if speed <= 0:
                    speed = self._get_base_speed(plane.plane_type)
            else:
                # Fallback to base speed from plane specs
                speed = self._get_base_speed(plane.plane_type)
            
            # Final safety check - ensure we have a valid speed
            if speed <= 0:
                print(f"Warning: Invalid speed {speed} for plane {plane.plane_type}, using default fallback")
                speed = 200.0  # Absolute fallback speed
            
            # Calculate flight hours for this leg
            if speed > 0 and distance > 0:
                leg_hours = distance / speed
                total_hours += leg_hours
            elif distance > 0:
                # If we have distance but no speed, something is wrong
                print(f"Warning: Distance {distance}nm but speed {speed}kts for plane {plane.plane_type}")
        
        # Calculate average speed
        if total_hours > 0:
            avg_speed = total_distance / total_hours
        else:
            # If no hours calculated, use base speed as fallback
            avg_speed = self._get_base_speed(plane.plane_type)
        
        return total_hours, avg_speed
    
    def _calculate_flight_hours(self, plane: PlaneInfo, legs: List[Dict[str, Any]]) -> float:
        """Calculate total flight hours for job legs (backward compatibility)."""
        flight_hours, _ = self._calculate_flight_hours_and_speed(plane, legs)
        return flight_hours
    
    def _calculate_route_info(self, conn: sqlite3.Connection, legs: List[Dict[str, Any]]) -> Tuple[str, Optional[int]]:
        """Calculate route string and minimum airport size from legs."""
        if not legs:
            return "", None
        
        # Build route string
        route_points = []
        min_airport_size = None
        
        for i, leg in enumerate(legs):
            from_icao = leg['from_icao']
            to_icao = leg['to_icao']
            
            # Add departure airport for first leg
            if i == 0 and from_icao:
                route_points.append(from_icao)
            
            # Add destination airport
            if to_icao:
                route_points.append(to_icao)
            
            # Check airport sizes
            for icao in [from_icao, to_icao]:
                if icao:
                    row = conn.execute("SELECT size FROM airports WHERE icao = ?", (icao,)).fetchone()
                    if row and row[0] is not None:
                        size = int(row[0])
                        min_airport_size = size if (min_airport_size is None or size < min_airport_size) else min_airport_size
        
        # Remove consecutive duplicates
        unique_route = []
        for point in route_points:
            if not unique_route or unique_route[-1] != point:
                unique_route.append(point)
        
        route_str = " -> ".join(unique_route)
        return route_str, min_airport_size
    
    def _get_base_speed(self, plane_type: str) -> float:
        """Get base cruise speed for a plane type."""
        with db_mod.connect(self.db_path) as conn:
            # First try plane_specs table (optimization data from planes.xlsx)
            row = conn.execute("SELECT cruise_speed_kts FROM plane_specs WHERE plane_type = ?", (plane_type,)).fetchone()
            if row and row[0]:
                return float(row[0])
            
            # Fallback to airplanes table (fetched API data)
            row = conn.execute("SELECT data_json FROM airplanes WHERE type = ? LIMIT 1", (plane_type,)).fetchone()
            if row and row[0]:
                try:
                    data = json.loads(row[0])
                    # Try AircraftType.designSpeedVC first (most accurate)
                    aircraft_type = data.get('AircraftType', {})
                    speed = aircraft_type.get('designSpeedVC')
                    if speed and speed > 0:
                        return float(speed)
                    
                    # Fallback to direct CruiseSpeed field if available
                    speed = data.get('CruiseSpeed')
                    if speed and speed > 0:
                        return float(speed)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        
        return 200.0  # Default fallback speed
    
    def _calculate_time_remaining_hours(self, conn: sqlite3.Connection, job_id: str) -> float:
        """Calculate time remaining until job expiration in hours."""
        row = conn.execute("SELECT data_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row or not row[0]:
            return 0.0
        
        try:
            job_data = json.loads(row[0])
            expiration_str = job_data.get("ExpirationDate")
            if not expiration_str:
                return 0.0
            
            # Parse the expiration date
            expiration_dt = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
            if expiration_dt.tzinfo is None:
                expiration_dt = expiration_dt.replace(tzinfo=timezone.utc)
            
            # Calculate time remaining
            now = datetime.now(timezone.utc)
            time_remaining = expiration_dt - now
            
            return time_remaining.total_seconds() / 3600.0
        except Exception:
            return 0.0
    
    def find_chainable_jobs(self, plane: PlaneInfo, current_location: str, 
                           remaining_hours: float, epsilon_hours: float = 0.5) -> List[JobInfo]:
        """Find jobs that can be chained from the current location."""
        chainable = []
        
        for job in plane.feasible_jobs:
            # Skip if job is already used
            if job.job_id in self.used_jobs:
                continue
            
            # Check if job departs from current location (use first leg)
            if not job.legs or job.legs[0]['from_icao'].upper() != current_location.upper():
                continue
            
            # Check if job fits within remaining time (with epsilon)
            if job.flight_hours > (remaining_hours + epsilon_hours):
                continue
            
            chainable.append(job)
        
        return chainable
    
    def find_jobs_from_major_hubs(self, plane: PlaneInfo, max_hours: float = 24.0) -> List[JobInfo]:
        """Find jobs from major job hubs that the plane could fly to first."""
        # Get top job departure airports
        with db_mod.connect(self.db_path) as conn:
            top_hubs = conn.execute("""
                SELECT from_icao, COUNT(DISTINCT job_id) as job_count 
                FROM job_legs 
                WHERE from_icao IS NOT NULL 
                GROUP BY from_icao 
                ORDER BY job_count DESC 
                LIMIT 20
            """).fetchall()
        
        hub_jobs = []
        for hub_icao, job_count in top_hubs:
            if hub_icao == plane.current_location:
                continue  # Skip if already at this hub
            
            # Find jobs from this hub that are feasible for this plane
            for job in plane.feasible_jobs:
                if job.job_id in self.used_jobs:
                    continue
                
                if not job.legs or job.legs[0]['from_icao'].upper() != hub_icao.upper():
                    continue
                
                # Add a small penalty for having to fly to the hub first
                # This will be factored into the job selection
                job.hub_penalty_hours = self._calculate_hub_transit_time(plane, plane.current_location, hub_icao)
                
                hub_jobs.append(job)
        
        return hub_jobs
    
    def _calculate_hub_transit_time(self, plane: PlaneInfo, from_icao: str, to_icao: str) -> float:
        """Calculate transit time from current location to a job hub."""
        # This is a simplified calculation - in reality you'd want to use actual distances
        # For now, assume a reasonable transit time based on plane type
        if plane.plane_type.startswith('Antonov'):
            return 2.0  # 2 hours transit for large cargo planes
        elif plane.plane_type.startswith('Rockwell'):
            return 1.5  # 1.5 hours for bombers
        else:
            return 1.0  # 1 hour for smaller planes
    
    def generate_work_order(self, plane: PlaneInfo, max_hours: float = 24.0, 
                           epsilon_hours: float = 0.5) -> WorkOrder:
        """Generate a work order for a single plane."""
        work_order = WorkOrder(
            plane_id=plane.plane_id,
            plane_registration=plane.registration,
            plane_type=plane.plane_type,
            priority=plane.priority
        )
        
        current_location = plane.current_location
        remaining_hours = max_hours
        hub_transit_added = False
        
        # First pass: greedy selection of best jobs
        while remaining_hours > 0:
            chainable_jobs = self.find_chainable_jobs(plane, current_location, remaining_hours, epsilon_hours)
            
            # If no local jobs found and we haven't tried hub jobs yet, look for hub jobs
            if not chainable_jobs and not hub_transit_added:
                hub_jobs = self.find_jobs_from_major_hubs(plane, remaining_hours)
                if hub_jobs:
                    # Find the best hub job
                    hub_jobs.sort(key=lambda j: work_order.get_priority_score(j), reverse=True)
                    best_hub_job = None
                    for job in hub_jobs:
                        # Account for transit time to hub
                        total_time = job.hub_penalty_hours + job.flight_hours
                        if total_time <= (remaining_hours + epsilon_hours):
                            best_hub_job = job
                            break
                    
                    if best_hub_job:
                        # Calculate actual distance for transit leg
                        transit_distance = self._calculate_distance_between_airports(
                            current_location, best_hub_job.legs[0]['from_icao']
                        )
                        
                        # Calculate actual flight hours for transit leg
                        transit_speed = self._get_base_speed(plane.plane_type)
                        transit_flight_hours = transit_distance / transit_speed if transit_speed > 0 else best_hub_job.hub_penalty_hours
                        
                        # Add transit time as a "virtual job" to account for flying to hub
                        transit_job = JobInfo(
                            job_id=f"transit_{plane.plane_id}_{best_hub_job.legs[0]['from_icao']}",
                            departure=current_location,
                            destination=best_hub_job.legs[0]['from_icao'],
                            pay=0.0,
                            xp=0.0,
                            distance_nm=transit_distance,
                            pay_per_hour=0.0,
                            xp_per_hour=0.0,
                            balance_score=0.0,
                            flight_hours=transit_flight_hours,
                            time_remaining_hours=999.0,  # Transit jobs don't expire
                            source="transit",
                            speed_kts=transit_speed,
                            route=f"{current_location} -> {best_hub_job.legs[0]['from_icao']}",
                            legs_count=0,  # Transit legs don't count as actual job legs
                            legs=[]  # Empty legs for transit jobs
                        )
                        
                        work_order.add_job(transit_job, work_order.total_hours)
                        current_location = best_hub_job.legs[0]['from_icao']
                        remaining_hours = max_hours - work_order.total_hours
                        hub_transit_added = True
                        continue
            
            if not chainable_jobs:
                break
            
            # Sort by priority score
            chainable_jobs.sort(key=lambda j: work_order.get_priority_score(j), reverse=True)
            
            # Take the best job that fits
            best_job = None
            for job in chainable_jobs:
                if work_order.can_add_job(job, max_hours, epsilon_hours):
                    best_job = job
                    break
            
            if not best_job:
                break
            
            # Add job to work order
            work_order.add_job(best_job, work_order.total_hours)
            
            # Mark job as used
            self.used_jobs.add(best_job.job_id)
            
            # Update location and remaining time
            current_location = best_job.destination
            remaining_hours = max_hours - work_order.total_hours
        
        # Second pass: try to fill remaining time with smaller jobs
        if remaining_hours > 0.1:  # Only if we have meaningful time left
            self._optimize_work_order_second_pass(work_order, plane, current_location, 
                                                 max_hours, epsilon_hours)
        
        # Optimize job order to minimize penalties while preserving chaining
        if len(work_order.jobs) > 1:
            self._optimize_job_order_for_penalties(work_order)
        
        return work_order
    
    def _optimize_work_order_second_pass(self, work_order: WorkOrder, plane: PlaneInfo,
                                        current_location: str, max_hours: float, 
                                        epsilon_hours: float) -> None:
        """Second pass optimization to fill remaining time."""
        remaining_hours = max_hours - work_order.total_hours
        
        while remaining_hours > 0.1:
            chainable_jobs = self.find_chainable_jobs(plane, current_location, remaining_hours, epsilon_hours)
            
            if not chainable_jobs:
                break
            
            # Find the best job that fits in remaining time
            best_job = None
            best_score = -1
            
            for job in chainable_jobs:
                if job.flight_hours <= (remaining_hours + epsilon_hours):
                    score = work_order.get_priority_score(job)
                    if score > best_score:
                        best_score = score
                        best_job = job
            
            if not best_job:
                break
            
            # Add the job
            work_order.add_job(best_job, work_order.total_hours)
            self.used_jobs.add(best_job.job_id)
            
            # Update for next iteration
            current_location = best_job.destination
            remaining_hours = max_hours - work_order.total_hours
    
    def _optimize_job_order_for_penalties(self, work_order: WorkOrder) -> None:
        """Optimize job order to minimize penalties while preserving chaining.
        
        Only sorts jobs that have the same departure and destination to avoid
        breaking the sequential chaining requirement.
        """
        if len(work_order.jobs) <= 1:
            return
        
        # Group consecutive jobs by departure/destination
        groups = []
        current_group = [work_order.jobs[0]]
        
        for i in range(1, len(work_order.jobs)):
            prev_job = work_order.jobs[i-1]
            curr_job = work_order.jobs[i]
            
            # Check if jobs have same departure and destination
            if (prev_job.departure == curr_job.departure and 
                prev_job.destination == curr_job.destination):
                current_group.append(curr_job)
            else:
                # End current group and start new one
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [curr_job]
        
        # Add the last group if it has multiple jobs
        if len(current_group) > 1:
            groups.append(current_group)
        
        # Sort each group by time remaining (ascending = most urgent first)
        for group in groups:
            group.sort(key=lambda j: j.time_remaining_hours)
        
        # Rebuild the job list maintaining the overall order
        new_jobs = []
        group_idx = 0
        current_group_idx = 0
        
        for i, job in enumerate(work_order.jobs):
            # Check if this job is part of a group that was sorted
            if group_idx < len(groups) and current_group_idx < len(groups[group_idx]):
                if job in groups[group_idx]:
                    # Use the sorted version from the group
                    new_jobs.append(groups[group_idx][current_group_idx])
                    current_group_idx += 1
                    
                    # Move to next group if current group is exhausted
                    if current_group_idx >= len(groups[group_idx]):
                        group_idx += 1
                        current_group_idx = 0
                else:
                    # Job not in a group, keep original order
                    new_jobs.append(job)
            else:
                # Job not in a group, keep original order
                new_jobs.append(job)
        
        work_order.jobs = new_jobs
        
        # Recalculate totals and penalties with correct accumulated times after reordering
        self._recalculate_work_order_totals(work_order)
    
    def _recalculate_work_order_totals(self, work_order: WorkOrder) -> None:
        """Recalculate work order totals after job reordering."""
        work_order.total_hours = 0.0
        work_order.total_pay = 0.0
        work_order.total_xp = 0.0
        work_order.total_adjusted_pay = 0.0
        
        accumulated_hours = 0.0
        for job in work_order.jobs:
            accumulated_hours += job.flight_hours
            
            # Recalculate penalty based on new position
            job.penalty_amount = work_order._calculate_penalty(job, accumulated_hours)
            job.adjusted_pay_total = job.pay - job.penalty_amount
            job.adjusted_pay_per_hour = job.adjusted_pay_total / job.flight_hours if job.flight_hours > 0 else 0.0
            
            work_order.total_hours += job.flight_hours
            work_order.total_pay += job.pay
            work_order.total_xp += job.xp
            work_order.total_adjusted_pay += job.adjusted_pay_total
    
    def generate_all_work_orders(self, max_hours: float = 24.0, 
                                epsilon_hours: float = 0.5) -> List[WorkOrder]:
        """Generate work orders for all idle planes."""
        idle_planes = self.get_idle_planes()
        work_orders = []
        
        print(f"Found {len(idle_planes)} idle planes")
        
        for i, plane in enumerate(idle_planes, 1):
            print(f"Processing plane {i}/{len(idle_planes)}: {plane.registration} ({plane.plane_type}, priority: {plane.priority})")
            
            # Load feasible jobs for this plane
            self.load_feasible_jobs_for_plane(plane)
            
            if not plane.feasible_jobs:
                print(f"  No feasible jobs found for {plane.registration}")
                continue
            
            print(f"  Found {len(plane.feasible_jobs)} feasible jobs")
            
            # Generate work order
            work_order = self.generate_work_order(plane, max_hours, epsilon_hours)
            
            if work_order.jobs:
                work_orders.append(work_order)
                print(f"  Generated work order with {len(work_order.jobs)} jobs, {work_order.total_hours:.2f} hours")
            else:
                print(f"  No suitable job chain found for {plane.registration}")
        
        return work_orders


def print_work_orders(work_orders: List[WorkOrder]) -> None:
    """Print work orders in a readable format."""
    if not work_orders:
        print("No work orders generated.")
        return
    
    print(f"\n{'='*80}")
    print(f"GENERATED {len(work_orders)} WORK ORDERS")
    print(f"{'='*80}")
    
    for i, wo in enumerate(work_orders, 1):
        print(f"\nWork Order #{i}")
        print(f"Plane: {wo.plane_registration} ({wo.plane_type})")
        print(f"Priority: {wo.priority.upper()}")
        print(f"Total Time: {wo.total_hours:.2f} hours")
        print(f"Total Pay: ${wo.total_pay:,.0f} (Adjusted: ${wo.total_adjusted_pay:,.0f})")
        print(f"Total XP: {wo.total_xp:,.0f}")
        print(f"Jobs: {len(wo.jobs)}")
        
        if wo.jobs:
            print("\nJob Sequence:")
            accumulated_hours = 0.0
            
            for j, job in enumerate(wo.jobs, 1):
                accumulated_hours += job.flight_hours
                late_indicator = f" [LATE {accumulated_hours - job.time_remaining_hours:.1f}h]" if accumulated_hours > job.time_remaining_hours else ""
                penalty_str = f" penalty=${job.penalty_amount:.0f}" if job.penalty_amount > 0 else ""
                
                print(f"  {j}. {job.departure} -> {job.destination}")
                print(f"     Job: {job.job_id}")
                print(f"     Distance: {job.distance_nm:.0f}nm, Time: {job.flight_hours:.2f}h")
                print(f"     Pay: ${job.pay:,.0f}/hr -> ${job.pay_per_hour:.0f}, XP: {job.xp_per_hour:.0f}/hr")
                print(f"     Time Remaining: {job.time_remaining_hours:.1f}h{late_indicator}{penalty_str}")
        
        print(f"\n{'-'*60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate work orders for idle planes")
    parser.add_argument("--max-hours", type=float, default=24.0, 
                       help="Maximum hours for work orders (default: 24.0)")
    parser.add_argument("--epsilon-hours", type=float, default=0.5,
                       help="Epsilon soft limit in hours (default: 0.5)")
    parser.add_argument("--csv", default="workorders.csv", 
                       help="Output CSV file path (default: workorders.csv)")
    parser.add_argument("--no-csv", action="store_true",
                       help="Skip CSV export")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize performance optimizer
    optimizer = None
    try:
        import os
        optimizer = PerformanceOptimizer(config.db_path)
        excel_path = os.path.join(os.path.dirname(__file__), 'planes.xlsx')
        if os.path.exists(excel_path):
            print("Loading performance optimization data...")
            optimizer.load_and_process(excel_path)
        else:
            print("No performance optimization data found, using default calculations")
            optimizer = None
    except Exception as e:
        print(f"Warning: Could not initialize performance optimizer: {e}")
        optimizer = None
    
    # Generate work orders
    generator = WorkOrderGenerator(config.db_path, optimizer)
    work_orders = generator.generate_all_work_orders(args.max_hours, args.epsilon_hours)
    
    # Print results
    print_work_orders(work_orders)
    
    # Export to CSV by default (unless --no-csv is specified)
    if not args.no_csv and work_orders:
        export_to_csv(work_orders, args.csv)
        print(f"\nWork orders exported to: {args.csv}")
    elif not work_orders:
        print("\nNo work orders generated.")
    
    return 0


def export_to_csv(work_orders: List[WorkOrder], csv_path: str) -> None:
    """Export work orders to CSV."""
    import csv
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'work_order_id', 'plane_id', 'plane_registration', 'plane_type', 'priority',
            'job_sequence', 'job_id', 'source', 'departure', 'destination', 'distance_nm',
            'flight_hours', 'pay', 'xp', 'pay_per_hour', 'xp_per_hour', 'balance_score',
            'time_remaining_hours', 'penalty_amount', 'adjusted_pay_per_hour', 'adjusted_pay_total',
            'speed_kts', 'min_airport_size', 'route', 'legs_count',
            'accumulated_hours', 'is_late'
        ])
        
        # Data
        for wo_id, wo in enumerate(work_orders, 1):
            accumulated_hours = 0.0
            
            for job_seq, job in enumerate(wo.jobs, 1):
                accumulated_hours += job.flight_hours
                is_late = accumulated_hours > job.time_remaining_hours
                
                writer.writerow([
                    wo_id, wo.plane_id, wo.plane_registration, wo.plane_type, wo.priority,
                    job_seq, job.job_id, job.source, job.departure, job.destination, job.distance_nm,
                    job.flight_hours, job.pay, job.xp, job.pay_per_hour, job.xp_per_hour, job.balance_score,
                    job.time_remaining_hours, job.penalty_amount, job.adjusted_pay_per_hour, job.adjusted_pay_total,
                    job.speed_kts, job.min_airport_size or "", job.route, job.legs_count,
                    accumulated_hours, is_late
                ])


if __name__ == "__main__":
    sys.exit(main())
