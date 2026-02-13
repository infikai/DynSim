# file: run.py

import pandas as pd
import argparse
from scheduler import Scheduler
from cluster_manager import ClusterManager
from components import Job
import time

# --- NEW: Define the target number of permanent LLM servers ---
TARGET_LLM_SERVERS = 10
TOTAL_GPUS = 1300

def load_jobs_from_csv(file_path):
    """
    Loads job definitions from a specified CSV file and counts job types.
    """
    print(f"Loading jobs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    jobs = []
    # ** NEW: Initialize counters **
    train_count = 0
    inference_count = 0

    df.rename(columns={
        'start_time_t': 'arrival_time',
        'runtime': 'base_duration',
        'gpu_wrk_util': 'utilization_required',
        'max_gpu_wrk_mem': 'memory_required'
    }, inplace=True)

    for index, row in df.iterrows():
        # Heuristic to determine job type
        if 'train' in str(row['task_group']).lower():
            job_type = 'training'
            train_count += 1 # Increment training counter
        else:
            job_type = 'inference'
            inference_count += 1 # Increment inference counter

        job = Job(id=f"job_{index}",
                  job_type=job_type,
                  base_duration=row['base_duration'],
                  arrival_time=row['arrival_time'],
                  memory_required=row['memory_required'],
                  utilization_required=row['utilization_required'])
        jobs.append(job)
        
    print(f"Successfully loaded {len(jobs)} total jobs.")
    # ** NEW: Print the counts **
    print(f"➡️ Found {train_count} training jobs and {inference_count} inference jobs in the workload.")
    return jobs

def load_llm_jobs_from_csv(file_path):
    """
    Loads LLM jobs from a large CSV file using a highly optimized approach.
    """
    print(f"Highly Optimized: Loading LLM jobs from {file_path}...")
    try:
        # Define the specific columns and their data types for efficiency.
        # This tells pandas exactly what to do, saving time and memory.
        job_spec = {
            'TIMESTAMP_seconds': 'float64',
            'ContextTokens': 'int32',
            'GeneratedTokens': 'int32'
        }

        # Use the 'pyarrow' engine for a significant speedup on large files.
        df = pd.read_csv(
            file_path,
            engine='pyarrow',
            usecols=job_spec.keys(),
            dtype=job_spec
        )

    except FileNotFoundError:
        print(f"Warning: The file '{file_path}' was not found. No LLM jobs will be loaded.")
        return []
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        print("Please ensure 'pyarrow' is installed (`pip install pyarrow`) and the CSV format is correct.")
        return []

    # Rename columns for use in the simulator.
    df.rename(columns={
        'TIMESTAMP_seconds': 'arrival_time',
        'ContextTokens': 'input_tokens',
        'GeneratedTokens': 'output_tokens'
    }, inplace=True)

    # The itertuples() loop is already fast; the main bottleneck was reading the file.
    jobs = [Job(id=f"llm_job_{row.Index}",
                job_type='llm_inference',
                arrival_time=row.arrival_time,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens)
            for row in df.itertuples()]

    print(f"➡️ Successfully loaded {len(jobs)} LLM inference jobs.")
    if jobs:
        print("\n--- Example LLM Job (after processing) ---")
        print(f"{jobs[0]!r}") # Using !r to call the __repr__ method
        print("-------------------------------------------\n")

    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced GPU Cluster Simulator")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the job workload.")
    parser.add_argument("--llm-trace", type=str, help="Path to the CSV file with LLM inference requests.")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Interval for printing progress to console.")
    parser.add_argument("--log-interval", type=int, default=1000, help="Interval for logging GPU usage to file.")
    parser.add_argument("--start-time", type=int, default=0, help="Simulation start time.")
    parser.add_argument("--end-time", type=int, default=-1, help="Simulation end time.")
    parser.add_argument("--tick-duration", type=int, default=1, help="The duration of each simulation time step (tick).")
    
    # ** NEW: Add argument for training job end-time threshold **
    parser.add_argument("--end-time-threshold", type=float, default=2, 
                        help="Multiplier for training job ideal duration (e.g., 1.2 means 20%% slack) for preemption.")
    
    args = parser.parse_args()

    simulation_start_time = time.time()
    print("🚀 Starting Simulation...")
    
    # --- MODIFIED: Create a single unified cluster ---
    print(f"Initializing a unified cluster with {TOTAL_GPUS} GPUs.")
    cluster = ClusterManager(num_gpus=TOTAL_GPUS, 
                             target_llm_servers=TARGET_LLM_SERVERS)
    
    # --- FINAL CHANGE: REMOVED pre-warming call ---
    print(f"LLM servers will be dynamically provisioned up to a soft target of {TARGET_LLM_SERVERS}.")
    
    job_workload = load_jobs_from_csv(args.csv_file)
    if args.llm_trace:
        llm_job_workload = load_llm_jobs_from_csv(args.llm_trace)
        job_workload.extend(llm_job_workload)
    
    read_job_time = time.time() - simulation_start_time
    print(f"\n Read job runtime: {read_job_time/60:.2f} minutes")
    
    # CRITICAL: Re-sort the combined list by arrival time
    job_workload.sort(key=lambda j: j.arrival_time)

    sorting_time = time.time() - simulation_start_time - read_job_time
    print(f"\n Sorting runtime: {sorting_time/60:.2f} minutes")
    
    if job_workload:
        scheduler = Scheduler(job_workload, cluster, 
                              progress_interval=args.progress_interval,
                              log_interval=args.log_interval,
                              start_time=args.start_time,
                              end_time=args.end_time,
                              tick_duration=args.tick_duration,
                              end_time_threshold=args.end_time_threshold) # Pass new arg
        scheduler.run_simulation()
        
        print("\n✅ Simulation Finished.")
        scheduler.print_results()
    else:
        print("Simulation aborted due to missing workload.")

    # <-- 3. Calculate and print the total elapsed time
    simulation_end_time = time.time()
    elapsed_time = simulation_end_time - simulation_start_time
    print(f"\n⏱️ Total real-world runtime: {elapsed_time/60/60:.2f} hours")