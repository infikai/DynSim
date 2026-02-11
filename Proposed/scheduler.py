# file: scheduler.py
import math
from collections import deque
from components import SimulationClock, PREEMPTION_OVERHEAD

INFERENCE_PREEMPTION_THRESHOLD = 300  # Threshold for inference preemption rights

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, progress_interval=1000, log_interval=500, 
                 start_time=0, end_time=-1, tick_duration=1, end_time_threshold=1.5):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0
        self.preemption_map = {} 
        self.end_time_threshold = end_time_threshold
        self.jobs_to_retry = deque()
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time
        
        # Initialize log files
        self.training_log_file = open('training_jobs.csv', 'w')
        self.usage_log_file = open('gpu_usage.csv', 'w')
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

        for gpu in self.cluster.unified_pool:
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

    def _batch_dispatch_llm_jobs(self, llm_jobs):
        remaining = deque(llm_jobs)
        while remaining:
            gpu = self.cluster.find_gpu_for_llm_job(self.clock.current_time)
            
            if not gpu:  # Try converting idle GPUs
                idle_pool = self.cluster.find_idle_gpus()
                if idle_pool:
                    gpu = idle_pool[0]
                    gpu.convert_to_llm_server()

            if not gpu:  # Try preempting training jobs if inference ≥250 GPUs
                inference_gpus_used = sum(len(j.assigned_gpus) for j in self.running_jobs 
                                          if j.job_type in ['inference', 'llm_inference'])
                
                if inference_gpus_used >= INFERENCE_PREEMPTION_THRESHOLD:
                    v_job, v_gpu = self.cluster.find_training_job_to_preempt(self.clock.current_time)
                    
                    if v_job:
                        if len(v_job.assigned_gpus) <= 1:
                            # 1-GPU job: checkpoint and re-queue
                            v_job.checkpoint_and_pause(v_gpu, self.clock.current_time)
                            self.running_jobs.remove(v_job)
                            self.jobs_to_retry.appendleft(v_job)
                        else:
                            # Multi-GPU job: borrow-and-return
                            v_job.preempt_and_pause(v_gpu, self.clock.current_time)
                            self.preemption_map[v_gpu.gpu_id] = v_job
                        self.preemption_count += 1
                        v_gpu.convert_to_llm_server(drain_at_time=self.clock.current_time + 600)
                        gpu = v_gpu
                        gpu.reclamation_cooldown_timer = PREEMPTION_OVERHEAD

            if gpu:
                while gpu.llm_slots_available > 0 and remaining:
                    job = remaining.popleft()
                    overhead = max(0, gpu.reclamation_cooldown_timer)
                    job.remaining_work += overhead
                    job.assign_resources([gpu], self.clock.current_time)
                    gpu.assign_llm_task(job)
                    self.running_jobs.append(job)
            else: 
                break
        return list(remaining)

    def _dispatch_training_job(self, job):
        job.gpus_needed = max(math.ceil(job.memory_required / 32), math.ceil(job.utilization_required / 100), 1)

        # Gather all idle or convertible resources
        immediate = []
        for g in self.cluster.find_idle_or_convertible_gpus():
            if g.is_llm_server and not g.running_tasks:
                g.revert_from_llm_server()
            immediate.append(g)

        if len(immediate) >= job.gpus_needed:
            if job.is_checkpointed:
                job.resume_from_checkpoint(immediate[0], self.clock.current_time)
            else:
                job.assign_resources(immediate[:job.gpus_needed], self.clock.current_time)
            self.running_jobs.append(job)
            return True
        return False

    def _handle_completion(self, job):
        """
        Processes a finished job, logs training data, and handles reclamation
        and LLM server switch-back.
        """
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job)
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        # Log training job completion
        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            log_entry = (f"{job.id},{job.arrival_time},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{job.gpus_needed}\n")
            self.training_log_file.write(log_entry)

        for gpu in freed_gpus:
            if gpu.gpu_id in self.preemption_map:
                reclaimer = self.preemption_map[gpu.gpu_id]
                if gpu.is_idle() or (gpu.is_llm_server and not gpu.running_tasks):
                    if gpu.is_llm_server: gpu.revert_from_llm_server()
                    del self.preemption_map[gpu.gpu_id]
                    if reclaimer in self.running_jobs:
                        reclaimer.reclaim_gpu(gpu, self.clock.current_time)
                        self.reclamation_count += 1

    def run_simulation(self):
        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            self.clock.tick()
            
            # 1. Collect new arrivals
            current_arrivals = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                current_arrivals.append(self.pending_jobs.popleft())

            # 2. Dispatch
            llm_batch = [j for j in current_arrivals if j.job_type == 'llm_inference']
            train_batch = [j for j in current_arrivals if j.job_type == 'training']
            
            unassigned_llm = self._batch_dispatch_llm_jobs(llm_batch)
            unassigned_train = [j for j in train_batch if not self._dispatch_training_job(j)]
            self.jobs_to_retry.extend(unassigned_llm + unassigned_train)

            # 3. Update Progress
            finished = []
            for job in self.running_jobs:
                job.update_progress(1, self.clock.current_time)
                if job.is_complete(): finished.append(job)
            
            for job in finished: self._handle_completion(job)
            
            # 4. Log GPU usage periodically
            if self.clock.current_time % self.log_interval == 0:
                self._log_gpu_usage()
            
            # 5. Timers
            for g in self.cluster.unified_pool:
                if g.reclamation_cooldown_timer > 0: g.reclamation_cooldown_timer -= 1
        
        # Close log files after simulation
        self.training_log_file.close()
        self.usage_log_file.close()
    
    def print_results(self):
        """Print simulation results and statistics."""
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        
        total_jobs = len(self.completed_jobs)
        training_jobs = [j for j in self.completed_jobs if j.job_type == 'training']
        inference_jobs = [j for j in self.completed_jobs if j.job_type == 'inference']
        llm_jobs = [j for j in self.completed_jobs if j.job_type == 'llm_inference']
        
        print(f"\n📊 Job Statistics:")
        print(f"   Total Jobs Completed: {total_jobs}")
        print(f"   - Training Jobs: {len(training_jobs)}")
        print(f"   - Inference Jobs: {len(inference_jobs)}")
        print(f"   - LLM Inference Jobs: {len(llm_jobs)}")
        
        print(f"\n⚡ Resource Management:")
        print(f"   Total Preemptions: {self.preemption_count}")
        print(f"   Total Reclamations: {self.reclamation_count}")
        
        if training_jobs:
            avg_training_tat = sum(j.turnaround_time for j in training_jobs) / len(training_jobs)
            print(f"\n🎯 Training Job Performance:")
            print(f"   Average Turnaround Time: {avg_training_tat:.2f}s")
        
        if inference_jobs:
            avg_inference_tat = sum(j.turnaround_time for j in inference_jobs) / len(inference_jobs)
            print(f"\n🎯 Inference Job Performance:")
            print(f"   Average Turnaround Time: {avg_inference_tat:.2f}s")
        
        if llm_jobs:
            avg_llm_tat = sum(j.turnaround_time for j in llm_jobs) / len(llm_jobs)
            print(f"\n🎯 LLM Inference Job Performance:")
            print(f"   Average Turnaround Time: {avg_llm_tat:.2f}s")
        
        print(f"\n⏱️  Simulation Time: {self.clock.current_time}s")
        print("="*60 + "\n")