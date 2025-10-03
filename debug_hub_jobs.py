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
            print(f'Job departure: {job.departure}')
            print(f'Job destination: {job.destination}')
            print(f'Job legs: {len(job.legs)}')
            if job.legs:
                print(f'First leg: {job.legs[0]["from_icao"]} -> {job.legs[0]["to_icao"]}')
            
            # Check if there are hub jobs available
            hub_jobs = generator.find_jobs_from_major_hubs(pano3, 24.0)
            print(f'Hub jobs available: {len(hub_jobs)}')
            if hub_jobs:
                print(f'First hub job: {hub_jobs[0].job_id} from {hub_jobs[0].legs[0]["from_icao"]}')

if __name__ == '__main__':
    main()
