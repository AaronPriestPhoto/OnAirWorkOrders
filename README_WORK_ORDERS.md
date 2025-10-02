# Work Order Generator for OnAir Airline Manager

This tool generates optimized work orders for idle planes by chaining together the most profitable jobs based on each plane's priority setting.

## Features

- **Intelligent Job Chaining**: Chains jobs sequentially from destination to departure airport
- **Time Constraints**: Respects configurable time limits (default 24 hours with 0.5hr epsilon soft limit)
- **Plane Prioritization**: Processes planes by efficiency: Pay > XP > Balanced, then by tail number
- **Job Exclusivity**: Ensures no job reuse across work orders (no two planes can fly the same job)
- **Penalty Optimization**: Sorts jobs within work orders by time remaining to minimize late penalties
- **Two-Pass Optimization**: Uses greedy selection followed by gap-filling for better time utilization
- **Performance Integration**: Uses optimized flight speeds from performance data when available

## Usage

### Basic Usage
```bash
python work_order_generator.py
```

### With Custom Time Limits
```bash
# 12-hour work orders with 1-hour epsilon
python work_order_generator.py --max-hours 12 --epsilon-hours 1.0
```

### Export to CSV
```bash
python work_order_generator.py --csv work_orders.csv
```

### Command Line Options

- `--max-hours HOURS`: Maximum hours for work orders (default: 24.0)
- `--epsilon-hours HOURS`: Epsilon soft limit in hours (default: 0.5)
- `--csv FILE`: Output CSV file path

## How It Works

### 1. Plane Selection
- Identifies idle planes (AircraftStatus = 3)
- Finds current location using coordinates if not explicitly set
- Sorts by priority: Pay planes first, then XP, then Balanced
- Within each priority group, sorts by registration/tail number

### 2. Job Chaining Algorithm
- For each plane, loads all feasible jobs from the job scoring system
- Starts from the plane's current location
- Uses greedy selection to pick the highest-scoring job that:
  - Departs from the current location
  - Fits within remaining time constraints
- Chains jobs sequentially (destination of job N becomes departure for job N+1)
- Applies two-pass optimization to fill remaining time

### 3. Penalty Calculation
- Calculates late penalties based on job expiration times
- Considers accumulated flight time for jobs later in the sequence
- Uses linear penalty scaling (0% at deadline, 100% at 24 hours late)
- Sorts jobs by time remaining to minimize total penalties

### 4. Priority Scoring
Each plane type has a priority setting that determines job selection:
- **Pay**: Maximizes adjusted pay per hour (after penalties)
- **XP**: Maximizes experience points per hour
- **Balance**: Maximizes balanced score (combination of pay and XP)

## Output Format

### Console Output
Shows detailed work orders with:
- Plane information (registration, type, priority)
- Total time, pay, XP for the work order
- Job sequence with routes, times, and penalty information

### CSV Output
Includes columns for:
- Work order and plane identification
- Job sequence and details
- Financial metrics (pay, penalties, adjusted pay)
- Timing information (flight hours, time remaining, late indicators)

## Example Output

```
Work Order #1
Plane: ANT-03 (Antonov AN-225-210)
Priority: BALANCE
Total Time: 22.09 hours
Total Pay: $492,700,601 (Adjusted: $465,617,521)
Total XP: 551
Jobs: 2

Job Sequence:
  1. PANC -> PANC
     Job: a843d2cc-2544-47a1-916e-c3eb698b4bc5
     Distance: 5982nm, Time: 11.96h
     Pay: $261,631,978/hr -> $21846615, XP: 25/hr
     Time Remaining: 7.0h [LATE 5.0h] penalty=$27083080
  2. PANC -> PANC
     Job: 463a9a00-4e5e-42d6-b1a3-b65ae6d94c61
     Distance: 5066nm, Time: 10.14h
     Pay: $231,068,624/hr -> $22658035, XP: 25/hr
     Time Remaining: 117.1h
```

## Prerequisites

- Completed job scoring (run `score_jobs.py` first)
- Plane specifications loaded (`load_plane_specs.py`)
- Performance optimization data (optional, `planes.xlsx`)
- Idle planes in your fleet (AircraftStatus = 3)

## Integration

The work order generator integrates with the existing OnAir toolkit:
- Uses the same database and configuration system
- Leverages job scoring results from `score_jobs.py`
- Incorporates performance optimization from `performance_optimizer.py`
- Respects plane priorities from the plane specifications

## Tips for Best Results

1. **Update Job Data**: Run `fetch_jobs.py` to get fresh job listings
2. **Score Jobs**: Run `score_jobs.py` to ensure feasibility calculations are current
3. **Set Plane Priorities**: Use `load_plane_specs.py` to configure plane priorities
4. **Performance Data**: Include `planes.xlsx` for optimized flight time calculations
5. **Regular Updates**: Re-run periodically as job availability changes

## Troubleshooting

- **No work orders generated**: Check that you have idle planes and feasible jobs
- **Jobs don't chain**: Verify jobs exist departing from your planes' current locations
- **Poor optimization**: Ensure plane priorities are set correctly in plane_specs table
- **Performance issues**: Large fleets may take time to process; consider filtering by plane type
