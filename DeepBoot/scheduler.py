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
        self.preemption_map = {}  # gpu_id -> training_job (borrow-and-return tracking)
        self.pending_return_events = []  # GPU return-to-training events with 140s init delay

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
            elif gpu.state == 'RUN':
                inference_gpus_used += 1
            elif gpu.state == 'TRAIN':
                borrowed_gpus_used += 1
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used},{protect_gpus_count}\n")
    
    def _can_preempt_without_deadline_miss(self, job, gpu, current_time):
        """
        Check if preempting this job would cause it to miss its deadline.
        Uses a 600-second estimated inference borrow time for calculations.
        
        Returns True if safe to preempt, False if would miss deadline.
        """
        ESTIMATED_BORROW_TIME = 600  # Expected inference running time on the borrowed GPU

        if job.id not in self.deadline_map:
            return True  # No deadline constraint, allow preemption
        
        deadline = self.deadline_map[job.id]
        
        remaining_work = job.remaining_work
        current_gpus = len(job.assigned_gpus)
        gpus_after_preempt = current_gpus - 1
        
        if gpus_after_preempt <= 0:
            # 1-GPU job: would be fully paused for ~600s + 140s init, then resume
            pause_penalty = ESTIMATED_BORROW_TIME + 140 + OVERHEAD_AFE_SYNC
            speedup = job.calculate_speedup(1)
            estimated_remaining_time = remaining_work / speedup if speedup > 0 else float('inf')
            estimated_completion = current_time + pause_penalty + estimated_remaining_time
            return estimated_completion <= deadline
        
        # Multi-GPU job: runs slower with one fewer GPU for ~600s, then gets it back
        speedup_reduced = job.calculate_speedup(gpus_after_preempt)
        speedup_full = job.calculate_speedup(current_gpus)
        
        # Work done during the borrow period at reduced speed
        work_during_borrow = ESTIMATED_BORROW_TIME * speedup_reduced
        work_after_borrow = remaining_work - work_during_borrow
        
        if work_after_borrow <= 0:
            # Job finishes during the borrow period
            estimated_remaining_time = remaining_work / speedup_reduced if speedup_reduced > 0 else float('inf')
        else:
            # Borrow period + init_time before GPU returns + remaining at full speed
            init_time = 140
            estimated_remaining_time = ESTIMATED_BORROW_TIME + init_time + (work_after_borrow / speedup_full if speedup_full > 0 else float('inf'))
        
        estimated_completion = current_time + estimated_remaining_time + OVERHEAD_AFE_SYNC
        return estimated_completion <= deadline
    
    def _dispatch_job(self, job):
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        return False
    
    def _assign_jobs_to_gpu(self, gpu, jobs_deque, base_overhead, max_slots=None):
        """Helper: assign as many jobs from jobs_deque to gpu as slots allow.
        Skips GPUs that are draining (past drain_at_time).
        Returns the number of jobs assigned."""
        # Don't assign new LLM tasks to a GPU that is draining back to training
        if gpu.is_draining(self.clock.current_time):
            return 0
        
        slots = max_slots if max_slots is not None else gpu.llm_slots_available
        count = 0
        
        while slots > 0 and jobs_deque:
            job = jobs_deque.popleft()
            job.assigned_gpus = [gpu]
            
            wait_time = gpu.reclaim_time_remaining
            job.start_time = self.clock.current_time + wait_time
            job.overhead_remaining = base_overhead

            delay = math.floor(max(0, job.start_time - job.arrival_time))
            if delay > 0:
                self.current_inference_delays.append(delay)
            
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            count += 1
            slots -= 1
        
        return count

    def _batch_dispatch_llm_jobs(self, llm_jobs):
        if not llm_jobs:
            return []

        remaining = deque(llm_jobs)

        # === Phase 1: RUN GPUs (active LLM servers with open slots) ===
        for gpu in self.cluster.inference_gpus:
            if not remaining:
                break
            if gpu.state == 'RUN' and gpu.llm_slots_available > 0 and not gpu.is_draining(self.clock.current_time):
                self._assign_jobs_to_gpu(gpu, remaining, OVERHEAD_WARM_START)

        # === Phase 2: PROTECT GPUs (warm but idle, fast to reuse) ===
        for gpu in self.cluster.inference_gpus:
            if not remaining:
                break
            if gpu.state == 'PROTECT':
                gpu.usage_count += 1
                gpu.state = 'RUN'
                self._assign_jobs_to_gpu(gpu, remaining, OVERHEAD_WARM_START)

        # === Phase 3: Reclaim from Training (preemption) ===
        while remaining:
            gpu, deadline_blocked = self.cluster.find_reclaim_target(
                can_preempt_fn=lambda j, g: self._can_preempt_without_deadline_miss(
                    j, g, self.clock.current_time
                )
            )
            
            if not gpu:
                if deadline_blocked > 0:
                    self.preemption_blocked_count += deadline_blocked
                break  # No preemptable targets, fall through to Phase 4

            # 1. Preempt Training Job
            if gpu.running_tasks:
                training_job = list(gpu.running_tasks.values())[0]['job']
                self._preempt_training_job(training_job, gpu)
                self.preemption_map[gpu.gpu_id] = training_job

            # 2. Start Physical Reclaim Timer with 600s drain
            gpu.start_reclaim(drain_at_time=self.clock.current_time + 600)
            self.preemption_count += 1

            # 3. Assign Leader + fill remaining slots
            leader = remaining.popleft()
            leader.start_time = self.clock.current_time + OVERHEAD_RECLAIM
            leader.overhead_remaining = OVERHEAD_WARM_START
            leader.assigned_gpus = [gpu]
            
            delay = math.floor(max(0, leader.start_time - leader.arrival_time))
            if delay > 0:
                self.current_inference_delays.append(delay)

            gpu.assign_llm_task(leader)
            self.running_jobs.append(leader)

            # Fill remaining slots on this GPU
            slots_remaining = LLM_MAX_CONCURRENCY - 1
            while slots_remaining > 0 and remaining:
                follower = remaining.popleft()
                follower.start_time = self.clock.current_time + OVERHEAD_RECLAIM
                follower.overhead_remaining = OVERHEAD_WARM_START
                follower.assigned_gpus = [gpu]
                
                delay_f = math.floor(max(0, follower.start_time - follower.arrival_time))
                if delay_f > 0:
                    self.current_inference_delays.append(delay_f)
                
                gpu.assign_llm_task(follower)
                self.running_jobs.append(follower)
                slots_remaining -= 1

        # === Phase 4: FREE GPUs (cold start, last resort) ===
        for gpu in self.cluster.inference_gpus:
            if not remaining:
                break
            if gpu.is_idle():  # state == 'FREE' and no tasks
                was_converted = gpu.convert_to_llm_server()
                if not was_converted:
                    continue
                self._assign_jobs_to_gpu(gpu, remaining, OVERHEAD_COLD_START)

        # Anything still remaining could not be assigned
        return list(remaining)

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
                    gpu.protect_time_remaining = 0
                gpu.assign_task(job)
                
            self.running_jobs.append(job)
            return True
        return False


    def _process_return_events(self):
        """Process pending GPU return-to-training events (140s init delay)."""
        for event in list(self.pending_return_events):
            event['time_left'] -= self.clock.tick_duration
            
            if event['time_left'] <= 0:
                job = event['job']
                gpu = event['gpu']
                
                if job not in self.running_jobs:
                    # Training job already finished, GPU goes FREE
                    gpu.state = 'FREE'
                    gpu.usage_count = 0
                    self.pending_return_events.remove(event)
                    continue
                
                # Return GPU to training job
                gpu.assign_task(job)
                job.assigned_gpus.append(gpu)
                job.overhead_remaining += OVERHEAD_AFE_SYNC
                self.pending_return_events.remove(event)

    def _handle_job_completion(self, job):

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
                        # Check if this GPU should return to a preempted training job
                        if gpu.gpu_id in self.preemption_map:
                            reclaimer = self.preemption_map.pop(gpu.gpu_id)
                            gpu.drain_at_time = -1
                            # Schedule GPU return with 140s init delay
                            if reclaimer in self.running_jobs:
                                gpu.state = 'TRAIN'  # Mark as TRAIN so it's not stolen during init
                                gpu.usage_count = 0
                                self.pending_return_events.append({
                                    'job': reclaimer,
                                    'gpu': gpu,
                                    'time_left': 140
                                })
                                self.reclamation_count += 1
                                # Compensate: put one FREE inference GPU into PROTECT
                                for inf_gpu in self.cluster.inference_gpus:
                                    if inf_gpu.is_idle():
                                        inf_gpu.state = 'PROTECT'
                                        inf_gpu.protect_time_remaining = inf_gpu.calculate_protection_time()
                                        break
                            else:
                                # Training job already finished, go to PROTECT
                                gpu.state = 'PROTECT'
                                gpu.protect_time_remaining = gpu.calculate_protection_time()
                        else:
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

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry or self.pending_return_events:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                break
            
            self.clock.tick()
            
            for gpu in self.cluster.inference_gpus:
                gpu.update_lifecycle(self.clock.tick_duration)

            self._process_return_events()

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

        self.print_results()

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