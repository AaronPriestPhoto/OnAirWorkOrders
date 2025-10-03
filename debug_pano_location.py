#!/usr/bin/env python3

from work_order_generator import WorkOrderGenerator
from onair.config import load_config
from performance_optimizer import PerformanceOptimizer

def main():
    config = load_config()
    optimizer = PerformanceOptimizer(config.db_path)
    generator = WorkOrderGenerator(config.db_path, optimizer)

    # Get PANO-3
    planes = generator.get_idle_planes()
    pano3 = None
    for plane in planes:
        if 'PANO-3' in plane.registration:
            pano3 = plane
            break

    if pano3:
        print(f'PANO-3 current location: {pano3.current_location}')
        generator.load_feasible_jobs_for_plane(pano3)
        if pano3.feasible_jobs:
            job = pano3.feasible_jobs[0]
            print(f'Job ID: {job.job_id}')
            print(f'Job departure: {job.departure}')
            print(f'Job legs: {len(job.legs)}')
            if job.legs:
                print(f'First leg from: {job.legs[0]["from_icao"]}')
                print(f'First leg to: {job.legs[0]["to_icao"]}')
            print(f'Job destination: {job.destination}')
            
            # Check if the job departs from current location
            if job.legs and job.legs[0]['from_icao']:
                first_leg_from = job.legs[0]['from_icao'].upper()
                current_location = pano3.current_location.upper()
                print(f'First leg from (upper): {first_leg_from}')
                print(f'Current location (upper): {current_location}')
                print(f'Match: {first_leg_from == current_location}')

if __name__ == '__main__':
    main()
