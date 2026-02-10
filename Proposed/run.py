# file: run_simulation.py

import pandas as pd
import argparse
from scheduler import Scheduler
from cluster_manager import ClusterManager
from components import Job
import time

# --- NEW: Define Pool Sizes ---
TRAINING_POOL_SIZE = 600
INFERENCE_POOL_SIZE = 500

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
    # (Same as before, optimized loading)
    try:
        df = pd.read_csv(file_path, engine='pyarrow') # assuming pyarrow installed
    except:
        df = pd.read_csv(file_path)
        
    jobs = []
    for row in df.itertuples():
        jobs.append(Job(id=f"llm_{row.Index}", job_type='llm_inference',
                        arrival_time=getattr(row, 'TIMESTAMP_seconds', 0),
                        input_tokens=getattr(row, 'ContextTokens', 10),
                        output_tokens=getattr(row, 'GeneratedTokens', 10)))
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Workload CSV")
    parser.add_argument("--llm-trace", type=str, help="LLM Trace CSV")
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--end-time", type=int, default=-1)
    
    args = parser.parse_args()

    print("🚀 Starting Split-Pool Simulation...")
    
    # --- MODIFIED: Initialize Split Cluster ---
    cluster = ClusterManager(num_training_gpus=TRAINING_POOL_SIZE, 
                             num_inference_gpus=INFERENCE_POOL_SIZE)
    
    cluster.pre_warm_llm_servers()
    
    job_workload = load_jobs_from_csv(args.csv_file)
    if args.llm_trace:
        job_workload.extend(load_llm_jobs_from_csv(args.llm_trace))
    
    job_workload.sort(key=lambda j: j.arrival_time)

    scheduler = Scheduler(job_workload, cluster, 
                          progress_interval=1000, log_interval=500,
                          start_time=args.start_time, end_time=args.end_time,
                          tick_duration=1, end_time_threshold=1.5)
    
    scheduler.run_simulation()
    scheduler.print_results()