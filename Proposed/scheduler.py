# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        PREEMPTION_OVERHEAD, RECLAMATION_OVERHEAD, 
                        LLM_POLICY_INTERVAL, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN)

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration, end_time_threshold):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0
        
        # Maps GPU ID -> Job object waiting for it
        self.preemption_map = {} 

        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time
        self.end_time_threshold = end_time_threshold

        self.delay_log_interval = 600 
        self.next_delay_log_time = 0 
        self.current_inference_delays = [] 

        self.training_log_file = open("training_job_log.csv", "w")
        self.usage_log_file = open("gpu_usage_log.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,train_pool_used,infer_pool_used_by_train,infer_pool_active_llm,train_pool_active_llm\n")

    def _log_gpu_usage(self):
        train_pool_used = sum(1 for g in self.cluster.training_pool if not g.is_idle() and not g.is_llm_server)
        infer_pool_borrowed = sum(1 for g in self.cluster.inference_pool 
                                  if not g.is_llm_server and not g.is_idle())
        infer_pool_llm = sum(1 for g in self.cluster.inference_pool 
                             if g.is_llm_server and g.running_tasks)
        train_pool_llm = sum(1 for g in self.cluster.training_pool 
                             if g.is_llm_server and g.running_tasks)

        self.usage_log_file.write(f"{self.clock.current_time},{train_pool_used},{infer_pool_borrowed},{infer_pool_llm},{train_pool_llm}\n")

    def _log_average_inference_delay(self):
        if not self.current_inference_delays:
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)
        self.inference_delay_log_file.write(f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n")
        self.current_inference_delays = []

    def _dispatch_job(self, job):
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        elif job.job_type == 'inference':
             return self._dispatch_training_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        jobs_to_assign = deque(llm_jobs)
        
        while jobs_to_assign:
            gpu = None
            
            # P1: Existing Server with Slots (Inference Pool)
            gpu = self.cluster.find_gpu_for_llm_job(self.clock.current_time)

            # P2: Find idle Inference GPU (Convert and use)
            if not gpu:
                idle_infer_gpus = self.cluster.find_idle_gpus_in_inference_pool()
                if idle_infer_gpus:
                    gpu = idle_infer_gpus[0]
                    gpu.convert_to_llm_server()

            # P3: Borrow idle Training GPU (Convert and use)
            if not gpu:
                idle_train_gpus = self.cluster.find_idle_gpus_in_training_pool()
                if idle_train_gpus:
                    gpu = idle_train_gpus[0]
                    gpu.convert_to_llm_server()

            # P4: Reclaim from Training job squatting on Inference Pool
            if not gpu:
                victim_job, victim_gpu = self.cluster.find_borrowed_gpu_to_reclaim(self.clock.current_time)
                if victim_job and victim_gpu:
                    victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                    self.preemption_map[victim_gpu.gpu_id] = victim_job
                    self.preemption_count += 1
                    
                    victim_gpu.convert_to_llm_server(drain_at_time=self.clock.current_time + 1000)
                    gpu = victim_gpu
                    
                    # --- NEW: Set Cooldown Timer ---
                    gpu.reclamation_cooldown_timer = PREEMPTION_OVERHEAD

            # P5: Preempt from Training job in Training Pool
            if not gpu:
                victim_job, victim_gpu = self.cluster.find_training_job_to_preempt_in_training_pool(self.clock.current_time)
                if victim_job and victim_gpu:
                    victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                    self.preemption_map[victim_gpu.gpu_id] = victim_job
                    self.preemption_count += 1
                    
                    victim_gpu.convert_to_llm_server(drain_at_time=self.clock.current_time + 1000)
                    gpu = victim_gpu
                    
                    # --- NEW: Set Cooldown Timer ---
                    gpu.reclamation_cooldown_timer = PREEMPTION_OVERHEAD

            if gpu:
                slots_to_fill = gpu.llm_slots_available
                
                for _ in range(slots_to_fill):
                    if not jobs_to_assign:
                        break 

                    job = jobs_to_assign.popleft() 
                    
                    # --- NEW: Calculate Overhead ---
                    overhead = 0
                    if gpu.reclamation_cooldown_timer > 0:
                        overhead = gpu.reclamation_cooldown_timer
                        
                    if overhead > 0:
                        job.remaining_work += overhead
                    
                    job.assigned_gpus = [gpu]
                    job.start_time = self.clock.current_time
                    
                    # --- NEW: Delay includes overhead ---
                    delay = math.floor(max(0, job.start_time - job.arrival_time)) + overhead
                    if delay > 0:
                        self.current_inference_delays.append(delay)
                    
                    gpu.assign_llm_task(job)
                    self.running_jobs.append(job)
            else:
                break 

        return list(jobs_to_assign)

    def _dispatch_training_job(self, job):
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        job.max_allowable_duration = job.ideal_duration * self.end_time_threshold

        # --- 1. Find IMMEDIATE Resources ---
        
        # A: Idle GPUs in Training Pool (Prioritize Own Pool)
        train_idle = []
        for g in self.cluster.training_pool:
            if g.is_idle(): 
                train_idle.append(g)
            elif g.is_llm_server and not g.running_tasks:
                g.revert_from_llm_server()
                train_idle.append(g)
        
        # B: Idle GPUs in Inference Pool (Borrow)
        infer_borrow = []
        for g in self.cluster.find_idle_resources_in_inference_pool():
            if g.is_llm_server:
                g.revert_from_llm_server() 
            infer_borrow.append(g)
            
        # Combine Immediate (Prioritize TrainPool then InferPool)
        immediate_gpus = train_idle + infer_borrow
        
        # --- 2. Dispatch Logic (Only Full Assignment) ---
        total_available = len(immediate_gpus)
        
        if total_available >= gpus_needed:
            # We can satisfy the job entirely with immediate resources
            gpus_to_start = immediate_gpus[:gpus_needed]
            
            job.assign_resources(gpus_to_start, self.clock.current_time)
            self.running_jobs.append(job)
            return True
        
        # Job must wait in pending
        return False

    def _handle_job_completion(self, job):
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job)
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            delay = max(0, job.start_time - job.arrival_time) if job.start_time != -1 else 0
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{job.gpus_needed}\n")
            self.training_log_file.write(log_entry)

        # --- Cleanup & Reclamation Logic ---
        for gpu in freed_gpus:
            
            # 1. Handle Preemption/Reclamation (Job waiting for this specific GPU)
            if gpu.gpu_id in self.preemption_map:
                reclaiming_job = self.preemption_map[gpu.gpu_id]
                
                # If GPU is now free (idle or empty server)
                if gpu.is_idle() or (gpu.is_llm_server and not gpu.running_tasks):
                    
                    # Revert if it was a server
                    if gpu.is_llm_server:
                         gpu.revert_from_llm_server()
                    
                    # Remove from map
                    del self.preemption_map[gpu.gpu_id]
                    
                    # Give to job if it's already running (Partial start case)
                    if reclaiming_job in self.running_jobs:
                        reclaiming_job.reclaim_gpu(gpu, self.clock.current_time)
                        self.reclamation_count += 1
                        # print(f"✅ Clock {self.clock.current_time}: Job {reclaiming_job.id} reclaimed draining GPU {gpu.gpu_id}.")
                    else:
                        # If job is still pending (started with 0 GPUs), it just becomes idle
                        # and will be picked up in the next dispatch loop.
                        pass
            
            # 2. Auto-revert logic for Training Pool GPUs
            # If an LLM job finishes on a Training GPU, and no one is explicitly waiting (map checked above),
            # check if it should revert anyway to be available for general training.
            if job.job_type == 'llm_inference' and gpu.pool_type == 'training':
                if gpu.is_llm_server and not gpu.running_tasks:
                     gpu.revert_from_llm_server()
                     # print(f"Testing: Reverted Training GPU {gpu.gpu_id} from LLM server to Idle.")

    def run_simulation(self):
        if not self.pending_jobs: return

        effective_start_time = self.start_time if self.start_time > 0 else self.pending_jobs[0].arrival_time
        self.clock.current_time = effective_start_time
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        
        self.jobs_to_retry = deque()
        self.pending_jobs = deque([j for j in self.pending_jobs if j.arrival_time >= effective_start_time])

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                break
            
            self.clock.tick()
            
            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            if self.clock.current_time % self.log_interval == 0:
                self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0:
                 print(f"🕒 Clock {self.clock.current_time}: Running={len(self.running_jobs)}, Pending={len(self.pending_jobs)}")
            
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                # Use the fast batch dispatcher
                unassigned_llm = self._batch_dispatch_llm_jobs(arrived_llm_jobs)
                
                unassigned_other = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other.append(job)

                self.jobs_to_retry.extend(unassigned_llm + unassigned_other)
            
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)
            
            for g in self.cluster.inference_pool:
                if g.is_llm_server and g.reclamation_cooldown_timer > 0:
                    g.reclamation_cooldown_timer -= self.clock.tick_duration
            
            for g in self.cluster.training_pool:
                if g.is_llm_server and g.reclamation_cooldown_timer > 0:
                    g.reclamation_cooldown_timer -= self.clock.tick_duration


    def print_results(self):
        self._log_average_inference_delay()
        self.training_log_file.close()
        self.usage_log_file.close()
        self.inference_delay_log_file.close()
        print("Simulation Complete. Results saved.")

        with open("simulation_summary.txt", "w") as summary_file:
            total_jobs = len(self.completed_jobs)
            if total_jobs == 0:
                summary_text = "No jobs were completed in the simulation window."
                print(summary_text)
                summary_file.write(summary_text + "\n")
                return

            training_jobs = [j for j in self.completed_jobs if j.job_type == 'training']
            inference_jobs = [j for j in self.completed_jobs if j.job_type == 'inference']
            llm_inference_jobs = [j for j in self.completed_jobs if j.job_type == 'llm_inference']
            
            # Combine regular and LLM inference for the average
            all_inference_jobs = inference_jobs + llm_inference_jobs
            
            avg_training_turnaround = sum(j.turnaround_time for j in training_jobs) / len(training_jobs) if training_jobs else 0
            avg_inference_turnaround = sum(j.turnaround_time for j in all_inference_jobs) / len(all_inference_jobs) if all_inference_jobs else 0
            
            # --- (NEW: Calculate average delays) ---
            avg_training_delay = sum(j.start_time - j.arrival_time for j in training_jobs if j.start_time != -1) / len(training_jobs) if training_jobs else 0
            avg_inference_delay = sum(j.start_time - j.arrival_time for j in all_inference_jobs if j.start_time != -1) / len(all_inference_jobs) if all_inference_jobs else 0

            # Build the output lines
            lines = [
                "--- Simulation Results ---",
                # --- (MODIFIED: Updated log file name in message) ---
                f"Detailed logs saved to 'training_job_log.csv', 'inference_delay_log.csv', and 'gpu_usage_log.csv'",
                f"Total Jobs Completed: {total_jobs}",
                # --- (REMOVED preemption/reclamation lines) ---
                f"Total Preemptions: {self.preemption_count}",
                f"Total Successful Reclamations: {self.reclamation_count}",
                f"Average Training Job Turnaround: {avg_training_turnaround:.2f} seconds",
                f"Average Inference Job Turnaround: {avg_inference_turnaround:.2f} seconds",
                # --- (NEW: Report average delays) ---
                f"Average Training Job Delay (Queue Time): {avg_training_delay:.2f} seconds",
                f"Average Inference Job Delay (Queue Time): {avg_inference_delay:.2f} seconds",
                "--------------------------"
            ]

            # Print to console and write to the summary file
            for line in lines:
                print(line)
                summary_file.write(line + "\n")
                