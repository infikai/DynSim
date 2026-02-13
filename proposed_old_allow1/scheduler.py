# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        PREEMPTION_OVERHEAD, RECLAMATION_OVERHEAD, 
                        LLM_POLICY_INTERVAL, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN)

class Scheduler:
    """
    Manages job scheduling with constrained preemption for LLM jobs
    in a unified GPU cluster.
    """
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration, end_time_threshold):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0
        self.preemption_map = {} # Maps gpu_id -> preempted training job

        # Intervals
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time

        self.end_time_threshold = end_time_threshold

        # Initialize log files
        self.training_log_file = open("training_job_log.csv", "w")
        self.usage_log_file = open("gpu_usage_log.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        """Writes headers to the log files."""
        self.training_log_file.write("job_id,arrival_time,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,llm_servers_total,llm_servers_busy\n")

    def _log_gpu_usage(self):
        """
        Logs a snapshot of GPU usage in the unified pool.
        """
        training_gpus_used = set()
        inference_gpus_used = set() 
        llm_servers_count = 0
        llm_servers_busy = 0

        for gpu in self.cluster.gpus:
            if gpu.is_llm_server:
                llm_servers_count += 1
                if gpu.running_tasks:
                    llm_servers_busy += 1
            else:
                for task in gpu.running_tasks.values():
                    job_type = task['job'].job_type
                    if job_type == 'training':
                        training_gpus_used.add(gpu.gpu_id)
                    elif job_type == 'inference':
                        inference_gpus_used.add(gpu.gpu_id)

        self.usage_log_file.write(f"{self.clock.current_time},{len(training_gpus_used)},{len(inference_gpus_used)},{llm_servers_count},{llm_servers_busy}\n")

    def _dispatch_job(self, job):
        """Routes a job to the appropriate dispatch method based on its type."""
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'inference':
            return self._dispatch_inference_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        """
        Dispatches a batch of LLM jobs using the priority flow: P1 -> P2 (Knob) -> P3.
        """
        if not llm_jobs:
            return []

        jobs_to_assign = deque(llm_jobs)
        
        # --- Priority 1: Fill all available slots on existing LLM servers ---
        existing_servers = [gpu for gpu in self.cluster.gpus if gpu.is_llm_server]
        existing_servers.sort(key=lambda g: g.llm_slots_available)

        for gpu in existing_servers:
            if not gpu.is_available_for_llm(self.clock.current_time):
                continue 
                
            slots_to_fill = gpu.llm_slots_available
            for _ in range(slots_to_fill):
                if not jobs_to_assign: break
                job = jobs_to_assign.popleft()
                job.assigned_gpus = [gpu]
                job.start_time = self.clock.current_time
                gpu.assign_llm_task(job)
                self.running_jobs.append(job)
            if not jobs_to_assign: break

        if not jobs_to_assign:
            return [] 

        # --- REINSTATED KNOB LOGIC ---
        
        non_draining_servers_count = sum(1 for gpu in self.cluster.gpus 
                                         if gpu.is_llm_server and gpu.drain_at_time == -1)
        
        # Threshold set to 300 as explicitly observed/requested by the user.
        PREEMPTION_THRESHOLD = 300
        allow_preemption = non_draining_servers_count > PREEMPTION_THRESHOLD

        if allow_preemption:
            # --- Priority 2: Preempt "safe" training jobs ---
            print(f"    [LLM Dispatch] P1 full. Knob active (Servers: {non_draining_servers_count} > Threshold: {PREEMPTION_THRESHOLD}). Trying P2 (Preempt).")
            
            while jobs_to_assign:
                victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
                if not victim_job:
                    break 
                
                print(f"    [LLM Dispatch] P2: Preempting Job {victim_job.id} on GPU {victim_gpu.gpu_id}.")
                victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                self.preemption_map[victim_gpu.gpu_id] = victim_job
                self.preemption_count += 1
                
                drain_time = self.clock.current_time + 1000.0
                victim_gpu.convert_to_llm_server(drain_at_time=drain_time)
                
                slots_to_fill = victim_gpu.llm_slots_available
                for _ in range(slots_to_fill):
                    if not jobs_to_assign: break
                    job = jobs_to_assign.popleft()
                    job.assigned_gpus = [victim_gpu]
                    job.start_time = self.clock.current_time
                    victim_gpu.assign_llm_task(job) 
                    self.running_jobs.append(job)
        
        # --- END OF P2 (Knob) LOGIC ---
        
        if not jobs_to_assign:
            return []

        # --- Priority 3: Grab idle GPUs and convert them (Last Resort) ---
        # Note: This runs if P1 was full AND P2 was skipped (due to the knob) or exhausted.
        print(f"    [LLM Dispatch] P2 skipped or exhausted. {len(jobs_to_assign)} jobs remain. Trying P3 (Expand).")
        while jobs_to_assign:
            idle_gpus = self.cluster.find_idle_gpus(1)
            if not idle_gpus:
                break 
            
            gpu_to_convert = idle_gpus[0]
            print(f"    [LLM Dispatch] P3: Expanding pool with idle GPU {gpu_to_convert.gpu_id}.")
            
            gpu_to_convert.convert_to_llm_server() 
            
            slots_to_fill = gpu_to_convert.llm_slots_available
            for _ in range(slots_to_fill):
                if not jobs_to_assign: break
                job = jobs_to_assign.popleft()
                job.assigned_gpus = [gpu_to_convert]
                job.start_time = self.clock.current_time
                gpu_to_convert.assign_llm_task(job) 
                self.running_jobs.append(job)

        return list(jobs_to_assign)

    def _dispatch_llm_inference_job(self, job):
        """
        Schedules a *single* LLM inference request using the 3-priority policy.
        """
        # P1: Find existing server with slots
        gpu = self.cluster.find_gpu_for_llm_job(self.clock.current_time)
        
        # P2: If none, preempt a "safe" training job
        if not gpu:
            victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
            if victim_job and victim_gpu:
                victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                self.preemption_map[victim_gpu.gpu_id] = victim_job
                self.preemption_count += 1
                
                drain_time = self.clock.current_time + 1000.0
                victim_gpu.convert_to_llm_server(drain_at_time=drain_time)
                gpu = victim_gpu
        
        # P3: If still none, grab an idle GPU
        if not gpu:
            idle_gpus = self.cluster.find_idle_gpus(1)
            if idle_gpus:
                gpu = idle_gpus[0]
                gpu.convert_to_llm_server()
        
        if gpu:
            job.assigned_gpus = [gpu]
            job.start_time = self.clock.current_time
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
            
        return False 

    def _dispatch_inference_job(self, job):
        """Routes a non-LLM inference job."""
        is_large_job = (job.memory_required > GPU_MEMORY_GB or 
                        job.utilization_required > GPU_UTILIZATION_PERCENT)
        
        if is_large_job:
            return self._dispatch_large_inference_job(job)
        else:
            return self._dispatch_stackable_inference_job(job)

    def _dispatch_stackable_inference_job(self, job):
        """Schedules a small inference job. No preemption."""
        job.gpus_needed = 1 
        gpu = self.cluster.find_gpu_for_stackable_inference(job)
        if gpu:
            job.assign_resources([gpu], self.clock.current_time)
            self.running_jobs.append(job)
            return True
        
        return False

    def _dispatch_large_inference_job(self, job):
        """Schedules a large inference job. No preemption."""
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        
        allocated_gpus = self.cluster.find_idle_gpus(gpus_needed)
        
        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
        return False
        
    def _dispatch_training_job(self, job):
        """Schedules a training job, no extra GPUs, sets threshold."""
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed

        job.max_allowable_duration = job.ideal_duration * self.end_time_threshold

        allocated_gpus = self.cluster.find_idle_gpus(gpus_needed)
        
        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
            
        return False

    def _apply_scale_down_policy(self):
        """
        Checks for and reverts all idle P3 (expansion) servers
        if the total server count is above the target.
        """
        num_llm_servers = sum(1 for g in self.cluster.gpus if g.is_llm_server)
        
        if num_llm_servers <= self.cluster.target_llm_servers:
            return 

        num_to_revert = num_llm_servers - self.cluster.target_llm_servers
        if num_to_revert <= 0:
            return
            
        candidates = []
        for gpu in self.cluster.gpus:
            if (gpu.is_llm_server and 
                not gpu.running_tasks and 
                gpu.gpu_id not in self.preemption_map):
                
                candidates.append(gpu)

        if candidates:
            num_can_revert = min(len(candidates), num_to_revert)
            
            if num_can_revert > 0:
                print(f"    [LLM Scale-Down Policy] Found {num_llm_servers} servers (Target: {self.cluster.target_llm_servers}). Reverting {num_can_revert} idle expansion servers.")
                
                for i in range(num_can_revert):
                    gpu_to_revert = candidates[i]
                    gpu_to_revert.revert_from_llm_server()
         
    def _handle_job_completion(self, job):
        """
        Processes a finished job, logs training data, and handles reclamation
        and LLM server switch-back.
        """
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job)
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            log_entry = (f"{job.id},{job.arrival_time},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{job.gpus_needed}\n")
            self.training_log_file.write(log_entry)

        llm_server_became_empty = False

        for gpu in freed_gpus:
            # Check if this GPU was a P2 (preempted) server
            if gpu.gpu_id in self.preemption_map:
                # Check if it's empty
                if gpu.is_idle() or (gpu.is_llm_server and not gpu.running_tasks):
                    reclaiming_job = self.preemption_map.pop(gpu.gpu_id) 
                    
                    if reclaiming_job in self.running_jobs:
                        if gpu.is_llm_server:
                            gpu.revert_from_llm_server() 
                        
                        reclaiming_job.reclaim_gpu(gpu, self.clock.current_time)
                        self.reclamation_count += 1
                        print(f"✅ Clock {self.clock.current_time}: RECLAIMED GPU {gpu.gpu_id} for training job {reclaiming_job.id}.")
                    
                    # Zombie Cleanup: Original job is gone, revert the server.
                    elif gpu.is_llm_server:
                        print(f"    [LLM Zombie Cleanup] Reverting P2 server {gpu.gpu_id} (original job {reclaiming_job.id} is gone).")
                        gpu.revert_from_llm_server()
            
            # Check if this GPU was a P3 (expansion) server
            elif gpu.is_llm_server and not gpu.running_tasks:
                llm_server_became_empty = True

        if llm_server_became_empty:
            self._apply_scale_down_policy()
    
    def run_simulation(self):
        """Main simulation loop."""
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

        effective_start_time = self.start_time
        if self.start_time == 0:
            effective_start_time = self.pending_jobs[0].arrival_time
            print(f"No specific start time given. Fast-forwarding to first job arrival at time {effective_start_time}.")

        original_job_count = len(self.pending_jobs)
        filtered_list = [j for j in self.pending_jobs if j.arrival_time >= effective_start_time]
        self.pending_jobs = deque(filtered_list)
        print(f"Filtered out {original_job_count - len(self.pending_jobs)} jobs that arrived before effective start time {effective_start_time}.")
        
        if not self.pending_jobs: 
            print("No jobs to simulate in the specified time window.")
            self.print_results()
            return

        self.clock.current_time = effective_start_time
        self.jobs_to_retry = deque()

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                print(f"\n🛑 Simulation ended at specified end time: {self.end_time}")
                break
            
            self.clock.tick()

            if self.clock.current_time > 0:
                
                # Log GPU usage
                if self.clock.current_time % self.log_interval == 0:
                    self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0 and (self.running_jobs or self.pending_jobs or self.jobs_to_retry):
                
                num_llm_servers = sum(1 for gpu in self.cluster.gpus if gpu.is_llm_server)
                num_llm_jobs = sum(1 for job in self.running_jobs if job.job_type == 'llm_inference')
                
                print(f"🕒 Clock {self.clock.current_time}: "
                      f"Pending={len(self.pending_jobs)}, "
                      f"Retrying={len(self.jobs_to_retry)}, "
                      f"Running={len(self.running_jobs)} (LLM: {num_llm_jobs}), "
                      f"LLM Servers={num_llm_servers}, "
                      f"Completed={len(self.completed_jobs)}")
            
            # --- Fairer Dispatch Logic ---
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                unassigned_llm_jobs = self._batch_dispatch_llm_jobs(arrived_llm_jobs)

                unassigned_other_jobs = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other_jobs.append(job)

                all_unassigned = sorted(unassigned_llm_jobs + unassigned_other_jobs, key=lambda j: j.arrival_time)
                self.jobs_to_retry.extend(all_unassigned)
            
            # --- Job Progress and Completion Handling ---
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)


    def print_results(self):
        """Prints a final summary and saves it to simulation_summary.txt."""
        self.training_log_file.close()
        self.usage_log_file.close()

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
            
            avg_training_turnaround = sum(j.turnaround_time for j in training_jobs) / len(training_jobs) if training_jobs else 0
            avg_inference_turnaround = sum(j.turnaround_time for j in inference_jobs) / len(inference_jobs) if inference_jobs else 0
            avg_llm_turnaround = sum(j.turnaround_time for j in llm_inference_jobs) / len(llm_inference_jobs) if llm_inference_jobs else 0
            
            lines = [
                "--- Simulation Results ---",
                f"Detailed logs saved to 'training_job_log.csv' and 'gpu_usage_log.csv'",
                f"Total Jobs Completed: {total_jobs}",
                f"  - Training: {len(training_jobs)}",
                f"  - Inference (Regular): {len(inference_jobs)}",
                f"  - Inference (LLM): {len(llm_inference_jobs)}",
                f"Total Preemptions (for LLM jobs): {self.preemption_count}",
                f"Total Successful Reclamations: {self.reclamation_count}",
                f"Average Training Job Turnaround: {avg_training_turnaround:.2f} seconds",
                f"Average Inference (Regular) Job Turnaround: {avg_inference_turnaround:.2f} seconds",
                f"Average Inference (LLM) Job Turnaround: {avg_llm_turnaround:.2f} seconds",
                "--------------------------"
            ]

            for line in lines:
                print(line)
                summary_file.write(line + "\n")