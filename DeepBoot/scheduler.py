# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        LLM_MAX_CONCURRENCY, OVERHEAD_WARM_START, OVERHEAD_COLD_START, 
                        OVERHEAD_RECLAIM, OVERHEAD_AFE_SYNC)

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration, deadline_map=None):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0

        self.pending_scaling_events = []

        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time

        self.delay_log_interval = 600
        self.next_delay_log_time = 0
        self.current_inference_delays = []
        
        # Deadline-aware preemption
        self.deadline_map = deadline_map or {}
        self.deadline_violations = 0
        self.preemption_blocked_count = 0

        self.training_log_file = open("training_job_log.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log.csv", "w")
        self.usage_log_file = open("gpu_usage_log.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus,gpus_in_protect\n")

    def _log_gpu_usage(self):
        training_gpus_used = 0
        inference_gpus_used = 0
        borrowed_gpus_used = 0 
        protect_gpus_count = 0 

        for gpu in self.cluster.training_gpus:
            if not gpu.is_idle():
                training_gpus_used += 1

        for gpu in self.cluster.inference_gpus:
            if gpu.state == 'PROTECT':
                protect_gpus_count += 1

            if gpu.is_llm_server:
                inference_gpus_used += 1
            elif not gpu.is_idle():
                inference_gpus_used += 1
                for task in gpu.running_tasks.values():
                    if task['job'].job_type == 'training':
                         borrowed_gpus_used += 1
                         inference_gpus_used -= 1
                         break
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used},{protect_gpus_count}\n")
    
    def _can_preempt_without_deadline_miss(self, job, gpu, current_time):
        """
        Check if preempting this job would cause it to miss its deadline.
        
        Returns True if safe to preempt, False if would miss deadline.
        """
        if job.id not in self.deadline_map:
            return True  # No deadline constraint, allow preemption
        
        deadline = self.deadline_map[job.id]
        
        # Estimate remaining time if preempted
        remaining_work = job.remaining_work
        current_gpus = len(job.assigned_gpus)
        gpus_after_preempt = current_gpus - 1
        
        if gpus_after_preempt <= 0:
            # Would completely stop the job
            return False
        
        # Estimate completion time after preemption
        speedup_after = job.calculate_speedup(gpus_after_preempt)
        estimated_remaining_time = remaining_work / speedup_after if speedup_after > 0 else float('inf')
        estimated_completion = current_time + estimated_remaining_time + OVERHEAD_AFE_SYNC
        
        # Allow preemption only if estimated completion <= deadline
        return estimated_completion <= deadline
    
    def _dispatch_job(self, job):
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        if not llm_jobs:
            return []

        # === Phase 1: Try Warm/Cold GPUs ===
        num_jobs_to_assign = len(llm_jobs)
        available_gpus, _ = self.cluster.find_resources_for_llm_batch(num_jobs_to_assign)
        
        assigned_count = 0
        job_index = 0
        
        for gpu in available_gpus:
            if job_index >= num_jobs_to_assign:
                break 

            base_overhead = OVERHEAD_WARM_START
            if gpu.state == 'FREE':
                base_overhead = OVERHEAD_COLD_START

            slots_to_fill = 0
            if not gpu.is_llm_server:
                was_converted = gpu.convert_to_llm_server()
                if not was_converted:
                    continue 
                slots_to_fill = gpu.llm_slots_available
            else:
                slots_to_fill = gpu.llm_slots_available
                
            for _ in range(slots_to_fill):
                if job_index >= num_jobs_to_assign:
                    break 
                
                job = llm_jobs[job_index]
                job.assigned_gpus = [gpu]
                
                wait_time = gpu.reclaim_time_remaining
                job.start_time = self.clock.current_time + wait_time
                job.overhead_remaining = base_overhead

                delay = math.floor(max(0, job.start_time - job.arrival_time))
                if delay > 0:
                    self.current_inference_delays.append(delay)
                
                gpu.assign_llm_task(job) 
                self.running_jobs.append(job)
                
                assigned_count += 1
                job_index += 1

        # === Phase 2: Try Reclaiming (Preemption) ===
        if assigned_count < num_jobs_to_assign:
            remaining_jobs = deque(llm_jobs[assigned_count:])
            still_unassigned = []

            while remaining_jobs:
                # Get the first job ("Leader")
                job_leader = remaining_jobs.popleft()
                
                # NEW: Pass deadline check to find_reclaim_target
                gpu = self.cluster.find_reclaim_target(
                    can_preempt_fn=lambda j, g: self._can_preempt_without_deadline_miss(
                        j, g, self.clock.current_time
                    )
                )
                
                
                if gpu:
                    # 1. Preempt Training Job
                    if gpu.running_tasks:
                        training_job = list(gpu.running_tasks.values())[0]['job']
                        self._preempt_training_job(training_job, gpu)

                    # 2. Start Physical Reclaim Timer
                    gpu.start_reclaim()
                    self.preemption_count += 1

                    # 3. Assign the Leader Job
                    # Wait for reclaim (Delay) + Warm Start Overhead
                    job_leader.start_time = self.clock.current_time + OVERHEAD_RECLAIM
                    job_leader.overhead_remaining = OVERHEAD_WARM_START
                    job_leader.assigned_gpus = [gpu]
                    
                    delay = math.floor(max(0, job_leader.start_time - job_leader.arrival_time))
                    if delay > 0:
                        self.current_inference_delays.append(delay)

                    gpu.assign_llm_task(job_leader)
                    self.running_jobs.append(job_leader)

                    # --- FIX: FILL REMAINING SLOTS IMMEDIATELY ---
                    # We just paid a huge penalty to open this GPU. Fill it up!
                    slots_remaining = LLM_MAX_CONCURRENCY - 1
                    
                    while slots_remaining > 0 and remaining_jobs:
                        job_follower = remaining_jobs.popleft()
                        
                        # Followers share the EXACT SAME wait conditions as the leader
                        job_follower.start_time = self.clock.current_time + OVERHEAD_RECLAIM
                        job_follower.overhead_remaining = OVERHEAD_WARM_START
                        job_follower.assigned_gpus = [gpu]
                        
                        delay_f = math.floor(max(0, job_follower.start_time - job_follower.arrival_time))
                        if delay_f > 0:
                            self.current_inference_delays.append(delay_f)
                        
                        gpu.assign_llm_task(job_follower)
                        self.running_jobs.append(job_follower)
                        slots_remaining -= 1

                else:
                    # No reclaim targets left (or all blocked by deadlines)
                    self.preemption_blocked_count += 1
                    still_unassigned.append(job_leader)
                    still_unassigned.extend(remaining_jobs)
                    break
            
            return still_unassigned

        return []

    def _dispatch_llm_inference_job(self, job):
        gpu = self.cluster.find_gpu_for_llm_job()
        
        if gpu:
            base_overhead = OVERHEAD_WARM_START
            if gpu.state == 'FREE':
                base_overhead = OVERHEAD_COLD_START
                gpu.convert_to_llm_server()
            else:
                if gpu.state == 'PROTECT':
                    gpu.usage_count += 1

            job.assigned_gpus = [gpu]
            
            wait_time = gpu.reclaim_time_remaining
            job.start_time = self.clock.current_time + wait_time
            job.overhead_remaining = base_overhead

            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
        
        # NEW: Pass deadline check to find_reclaim_target
        gpu = self.cluster.find_reclaim_target(
            can_preempt_fn=lambda j, g: self._can_preempt_without_deadline_miss(
                j, g, self.clock.current_time
            )
        )
        if gpu:
            if gpu.running_tasks:
                training_job = list(gpu.running_tasks.values())[0]['job']
                self._preempt_training_job(training_job, gpu)

            gpu.start_reclaim()

            job.start_time = self.clock.current_time + OVERHEAD_RECLAIM
            job.overhead_remaining = OVERHEAD_WARM_START
            job.assigned_gpus = [gpu]
            
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
        
        # No GPU available and preemption blocked by deadlines
        self.preemption_blocked_count += 1
        return False

    def _preempt_training_job(self, job, gpu):
        gpu.release_task(job)
        if gpu in job.assigned_gpus:
            job.assigned_gpus.remove(gpu)
        
        # FIX: Check if we already applied the penalty at this timestamp
        # We use dynamic attribute assignment to track the last penalty time
        current_time = self.clock.current_time
        last_penalty_time = getattr(job, 'last_penalty_time', -1)
        
        if last_penalty_time != current_time:
            job.overhead_remaining += OVERHEAD_AFE_SYNC
            job.last_penalty_time = current_time


        
    def _dispatch_training_job(self, job):
        desired_gpus = job.gpus_needed
        min_gpus = 1 

        assigned_gpus = []

        # Priority 1: Dedicated Training GPUs
        for gpu in self.cluster.training_gpus:
            if gpu.is_idle():
                assigned_gpus.append(gpu)
                if len(assigned_gpus) == desired_gpus:
                    break
        
        # Priority 2: Borrow from Inference Pool
        if len(assigned_gpus) < desired_gpus:
            for gpu in self.cluster.inference_gpus:
                if gpu.is_idle():
                    assigned_gpus.append(gpu)
                    if len(assigned_gpus) == desired_gpus:
                        break
        
        if len(assigned_gpus) == desired_gpus:
            job.assign_resources(assigned_gpus, self.clock.current_time)
            for gpu in assigned_gpus:
                if gpu.gpu_type == 'inference':
                    gpu.state = 'TRAIN'
                    gpu.protect_time_remaining = 0
                gpu.assign_task(job)
                
            self.running_jobs.append(job)
            return True
        return False

    def _try_scale_up_training_jobs(self):
        # If no GPUs are idle, don't bother checking if jobs want to scale.
        # This saves millions of loop iterations when the cluster is full.
        
        # Check quickly if we have ANY free resources
        has_idle_training = any(g.is_idle() for g in self.cluster.training_gpus)
        has_idle_inference = any(g.is_idle() for g in self.cluster.inference_gpus)
        
        if not has_idle_training and not has_idle_inference:
            return

        for job in self.running_jobs:
            if job.job_type == 'training':
                
                is_already_scaling = any(evt['job'] == job for evt in self.pending_scaling_events)
                if is_already_scaling:
                    continue

                current_count = len(job.assigned_gpus)
                desired = job.gpus_needed
                
                if current_count < desired:
                    needed = 1
                    newly_acquired = []

                    for gpu in self.cluster.training_gpus:
                        if gpu.is_idle():
                            newly_acquired.append(gpu)
                            if len(newly_acquired) == needed: break
                    
                    if len(newly_acquired) < needed:
                        remaining = needed - len(newly_acquired)
                        for gpu in self.cluster.inference_gpus:
                            if gpu.is_idle():
                                newly_acquired.append(gpu)
                                if len(newly_acquired) >= remaining: break
                        if len(newly_acquired) > needed:
                            newly_acquired = newly_acquired[:needed]

                    if newly_acquired:
                        for gpu in newly_acquired:
                             if gpu.gpu_type == 'inference':
                                gpu.state = 'TRAIN'
                                gpu.protect_time_remaining = 0
                        
                        if desired > 16:
                            init_time = 140
                        else:
                            init_time = 40
                        
                        self.pending_scaling_events.append({
                            'job': job,
                            'new_gpus': newly_acquired,
                            'time_left': init_time
                        })

    def _process_scaling_events(self):
        for event in list(self.pending_scaling_events):
            event['time_left'] -= self.clock.tick_duration
            
            if event['time_left'] <= 0:
                job = event['job']
                new_gpus = event['new_gpus']
                
                if job not in self.running_jobs:
                    for gpu in new_gpus:
                        if gpu.gpu_type == 'inference':
                            gpu.state = 'FREE'
                    self.pending_scaling_events.remove(event)
                    continue

                old_gpus = list(job.assigned_gpus)
                for gpu in old_gpus: 
                    gpu.release_task(job)
                
                all_gpus = old_gpus + new_gpus
                job.assign_resources(all_gpus, job.start_time)
                
                for gpu in all_gpus:
                    if gpu.gpu_type == 'inference':
                        gpu.state = 'TRAIN'
                    gpu.assign_task(job)
                
                job.overhead_remaining += OVERHEAD_AFE_SYNC
                self.reclamation_count += 1
                self.pending_scaling_events.remove(event)

    def _handle_job_completion(self, job):
        for event in list(self.pending_scaling_events):
            if event['job'] == job:
                for gpu in event['new_gpus']:
                    if gpu.gpu_type == 'inference':
                        gpu.state = 'FREE'
                        gpu.usage_count = 0
                self.pending_scaling_events.remove(event)

        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job) 
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        if job.job_type == 'training':
            delay = max(0, job.start_time - job.arrival_time)
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{len(freed_gpus)}\n")
            self.training_log_file.write(log_entry)

        if job.job_type == 'llm_inference':
            for gpu in freed_gpus:
                if gpu.gpu_type == 'inference':
                    if len(gpu.running_tasks) == 0:
                        gpu.state = 'PROTECT'
                        gpu.protect_time_remaining = gpu.calculate_protection_time()
                    else:
                        gpu.state = 'RUN'
        
        if job.job_type == 'training':
            for gpu in freed_gpus:
                # ALL GPUs released by a training job.
                gpu.state = 'FREE'
                gpu.usage_count = 0
    
    def _log_average_inference_delay(self):
        if not self.current_inference_delays:
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)

        log_entry = f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n"
        self.inference_delay_log_file.write(log_entry)
        self.current_inference_delays = []

    def run_simulation(self):
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

        effective_start_time = self.start_time
        if self.start_time == 0:
            effective_start_time = self.pending_jobs[0].arrival_time
            print(f"Fast-forwarding to time {effective_start_time}.")

        while self.pending_jobs and self.pending_jobs[0].arrival_time < effective_start_time:
             self.pending_jobs.popleft()
        
        if not self.pending_jobs: 
            print("No jobs left in window.")
            self.print_results()
            return

        self.clock.current_time = effective_start_time
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        self.jobs_to_retry = deque()

        # 1. Initialize the timer for scaling
        next_scaling_check = self.clock.current_time

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                break
            
            self.clock.tick()
            
            for gpu in self.cluster.inference_gpus:
                gpu.update_lifecycle(self.clock.tick_duration)

            if self.clock.current_time >= next_scaling_check:
                self._try_scale_up_training_jobs()
                next_scaling_check = self.clock.current_time + 900

            self._process_scaling_events()

            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            if self.clock.current_time > 0 and self.clock.current_time % self.log_interval == 0:
                self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0:
                num_llm = sum(1 for j in self.running_jobs if j.job_type=='llm_inference')

                # NEW: Calculate total pending (Future + Backlog)
                future_jobs = len(self.pending_jobs)
                retry_jobs = len(self.jobs_to_retry)
                total_pending = future_jobs + retry_jobs
                
                print(f"Clock {self.clock.current_time}: Pending={total_pending} (Backlog={retry_jobs}), Running={len(self.running_jobs)} (LLM:{num_llm}), Completed={len(self.completed_jobs)}")
            
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                unassigned_llm = self._batch_dispatch_llm_jobs(arrived_llm_jobs)
                
                unassigned_other = []
                for job in other_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other.append(job)

                retry_list = unassigned_llm + unassigned_other
                retry_list.sort(key=lambda j: j.arrival_time)
                self.jobs_to_retry.extend(retry_list)
            
            finished = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete(): finished.append(job)
            
            for job in finished: self._handle_job_completion(job)

    def print_results(self):
        self._log_average_inference_delay()
        self.training_log_file.close()
        self.inference_delay_log_file.close()
        self.usage_log_file.close()
        print("Simulation Complete. Results saved.")
        print(f"Total Preemptions: {self.preemption_count}")
        print(f"Total Successful Reclamations: {self.reclamation_count}")
        print(f"Preemptions Blocked by Deadlines: {self.preemption_blocked_count}")
        
        if self.deadline_map:
            # Compare actual vs baseline completion times
            matched_jobs = [j for j in self.completed_jobs if j.id in self.deadline_map]
            violations = sum(1 for j in matched_jobs if j.completion_time > self.deadline_map[j.id])
            print(f"Jobs Meeting Deadline: {len(matched_jobs) - violations}/{len(matched_jobs)}")
            if violations > 0:
                print(f"WARNING: {violations} jobs missed their deadlines!")