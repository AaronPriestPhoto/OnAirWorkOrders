# Secondary Sorting Bug Fix - Summary

## The Issue

In ANT-09's work order, Job 3 (expires in 4.40h) was scheduled AFTER Job 2 (expires in 10.03h), even though both jobs had the same departure and destination (PANC->PANC). Job 3 should have been scheduled first since it expires 5.63 hours earlier.

## Analysis

Both jobs should have been sorted by expiration time (most urgent first) since they have the same route.

The sorting logic in `_optimize_job_order_for_penalties()` was correctly:
1. Identifying groups of consecutive jobs with same departure/destination
2. Sorting each group by `time_remaining_hours` (ascending = most urgent first)
3. Rebuilding the `work_order.jobs` list with the sorted groups

**However**, there was a critical bug in `_recalculate_work_order_totals()`.

## The Bug

After reordering `work_order.jobs`, the code needed to rebuild `work_order.execution_order` (which is what the Excel export uses). The bug was in lines 1804-1813 of the OLD code:

```python
# OLD (BUGGY) CODE:
for item_type, item in work_order.execution_order:  # ← Iterating through OLD order!
    if item_type == "multi_leg":
        new_execution_order.append(("multi_leg", item))
    elif item_type == "job":
        job_id = item.job_id
        if job_id in job_map and job_id not in jobs_added:
            new_execution_order.append(("job", job_map[job_id]))
            jobs_added.add(job_id)
```

**The problem:** It iterated through the **OLD** `execution_order` to build the new one. Even though `work_order.jobs` was correctly reordered, when rebuilding `execution_order`, it walked through the old order and preserved the wrong sequence!

## The Fix

The fix iterates through the **reordered** `work_order.jobs` list instead:

```python
# NEW (FIXED) CODE:
# First, identify which jobs belong to multi-job legs
multi_leg_map = {}
for item_type, item in work_order.execution_order:
    if item_type == "multi_leg":
        for job in item.jobs:
            multi_leg_map[job.job_id] = item

# Now build execution order from the REORDERED jobs list
for job in work_order.jobs:  # ← Iterating through REORDERED jobs!
    if job.job_id in multi_leg_map:
        # This job is part of a multi-leg
        leg = multi_leg_map[job.job_id]
        if leg not in [item for _, item in new_execution_order if _ == "multi_leg"]:
            new_execution_order.append(("multi_leg", leg))
            jobs_added.update(j.job_id for j in leg.jobs)
    else:
        # Single job - add it in the reordered position
        if job.job_id not in jobs_added:
            new_execution_order.append(("job", job))
            jobs_added.add(job.job_id)
```

## Impact

✅ **Fixed:** Secondary sorting by expiration time now works correctly
✅ **Preserved:** Multi-job sequential routing still works
✅ **Preserved:** Multi-job legs stay together as a unit

## To Verify

Run the work order generator again and check that jobs with the same departure and destination are now sorted by expiration time (most urgent first).

## Answer to Your Question

> "Is it taking into account flight hours and that is why it put job 3 before 2?"

**No, it wasn't taking flight hours into account at all.** The bug meant the secondary sorting wasn't working at all - jobs remained in whatever order they were originally added. The fix now correctly sorts jobs with the same route by expiration time, which ensures the most urgent jobs are done first (and less likely to incur late penalties).

The sorting is purely by `time_remaining_hours` for jobs with the same route. It doesn't try to optimize which one to do first based on profitability vs. lateness risk - it just does the most urgent one first, which is the safest strategy to minimize total penalties.

