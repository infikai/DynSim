# file: components.py

# --- Global Constants ---
GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
PREEMPTION_OVERHEAD = 2
RECLAMATION_OVERHEAD = 3
PREEMPTION_COOLDOWN = 600
LLM_POLICY_INTERVAL = 10 

# NEW: LLM Inference Performance Model Constants
LLM_BASE_TTFT = 2.5          
LLM_TKN_PER_INPUT = 0.005    
LLM_TPOT = 0.1               
LLM_MAX_CONCURRENCY = 16      

class SimulationClock:
    """A simple discrete-time simulation clock."""
    def __init__(self, tick_duration=1):
        self.current_time = 0
        self.tick_duration = tick_duration

    def tick(self):
        self.current_time += self.tick_duration

class GPU:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        
        self.total_memory = GPU_MEMORY_GB
        self.total_utilization = GPU_UTILIZATION_PERCENT
        self.available_memory = self.total_memory
        self.available_utilization = self.total_utilization
        
        self.is_llm_server = False
        self.llm_slots_total = 0
        self.llm_slots_available = 0
        
        # --- NEW: Add drain timer attribute ---
        self.drain_at_time = -1 
        
        self.running_tasks = {}

    def can_fit(self, job):
        return (job.memory_required <= self.available_memory and
                job.utilization_required <= self.available_utilization)

    def assign_task(self, job, mem_slice, util_slice):
        if mem_slice > self.available_memory or util_slice > self.available_utilization:
            print(f"\n‼️ WARNING: ASSIGNMENT FAILED ON GPU {self.gpu_id}")
            print(f"   GPU State : Available Mem={self.available_memory:.2f}, Available Util={self.available_utilization:.2f}")
            print(f"   Job Slice : Required Mem={mem_slice:.2f}, Required Util={util_slice:.2f}")
            print(f"   Full Job  : {job!r}\n")
            raise Exception(f"Resource slice for job {job.id}: mem:{mem_slice}, util:{util_slice} cannot fit on GPU {self.gpu_id}: {self.available_memory}: {self.available_utilization}.")
        self.available_memory -= mem_slice
        self.available_utilization -= util_slice
        self.running_tasks[job.id] = {'job': job, 'mem': mem_slice, 'util': util_slice}

    def convert_to_llm_server(self, drain_at_time=-1): 
        if self.is_llm_server or not self.is_idle():
            return False
        
        self.is_llm_server = True
        self.llm_slots_total = LLM_MAX_CONCURRENCY
        self.llm_slots_available = LLM_MAX_CONCURRENCY
        
        self.drain_at_time = drain_at_time
        
        self.available_memory = 0
        self.available_utilization = 0
        return True

    def revert_from_llm_server(self):
        if not self.is_llm_server or self.running_tasks:
            return False
        
        self.is_llm_server = False
        self.llm_slots_total = 0
        self.llm_slots_available = 0
        
        self.drain_at_time = -1
        
        self.available_memory = self.total_memory
        self.available_utilization = self.total_utilization
        return True
    
    def is_available_for_llm(self, current_time):
        """
        Checks if this GPU can accept a *new* LLM job.
        """
        if not self.is_llm_server or self.llm_slots_available <= 0:
            return False
        
        # If a drain time is set, check if we are past it (it's draining).
        if self.drain_at_time != -1 and current_time >= self.drain_at_time:
            return False 
            
        return True 

    def assign_llm_task(self, job):
        if not self.is_llm_server or self.llm_slots_available <= 0:
            raise Exception(f"Attempted to assign LLM job to non-server or full GPU {self.gpu_id}")
        self.llm_slots_available -= 1
        self.running_tasks[job.id] = {'job': job, 'type': 'llm_inference'}

    def release_task(self, job):
        if job.id not in self.running_tasks: return
        task_info = self.running_tasks.pop(job.id)
        if task_info.get('type') == 'llm_inference':
            self.llm_slots_available += 1
        else: # Regular job
            self.available_memory += task_info['mem']
            self.available_utilization += task_info['util']
    
    def is_idle(self):
        """
        A GPU is only truly idle if it has no tasks AND it is not currently
        configured as an exclusive-use LLM server.
        """
        return not self.running_tasks and not self.is_llm_server

    def get_running_training_jobs(self):
        return [task['job'] for task in self.running_tasks.values() if task['job'].job_type == 'training']

class Job:
    def __init__(self, id, job_type, arrival_time, base_duration=0, 
                 memory_required=0, utilization_required=0, 
                 input_tokens=0, output_tokens=0):
        self.id = id
        self.job_type = job_type
        self.base_duration = base_duration
        self.arrival_time = arrival_time
        self.memory_required = max(memory_required, 1)
        self.utilization_required = max(utilization_required, 1)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        # If it's an LLM job, calculate its duration now
        if self.job_type == 'llm_inference':
            time_to_process_input = LLM_TKN_PER_INPUT * self.input_tokens
            time_to_generate_output = LLM_TPOT * self.output_tokens
            self.base_duration = LLM_BASE_TTFT + time_to_process_input + time_to_generate_output
        
        self.remaining_work = self.base_duration
        self.assigned_gpus = []
        self.start_time = -1
        self.completion_time = -1
        self.turnaround_time = -1
        self.paused_until = -1
        self.gpus_needed = 1
        self.last_preemption_time = -1
        
        # --- NEW: Tracking for end-time threshold ---
        self.ideal_duration = self.base_duration
        self.max_allowable_duration = float('inf') 
        
    def __repr__(self):
        return (f"<Job id={self.id} type={self.job_type} "
                f"mem_req={self.memory_required:.2f} util_req={self.utilization_required:.2f} "
                f"in_token={self.input_tokens} out_token={self.output_tokens} "
                f"gpus_needed={self.gpus_needed} duration={self.base_duration} remain={self.remaining_work}>")

    def assign_resources(self, gpus, current_time):
        self.assigned_gpus = gpus
        self.start_time = current_time
        self._distribute_load()

    def _distribute_load(self):
        num_gpus_originally_needed = self.gpus_needed
        
        if num_gpus_originally_needed == 0:
            if len(self.assigned_gpus) > 0:
                num_gpus_originally_needed = len(self.assigned_gpus)
            else:
                return 

        mem_per_gpu = self.memory_required / num_gpus_originally_needed
        util_per_gpu = self.utilization_required / num_gpus_originally_needed
        
        for gpu in self.assigned_gpus:
            gpu.assign_task(self, mem_per_gpu, util_per_gpu)

    def preempt_and_pause(self, gpu_to_release, current_time):
        """Handles the logic of being preempted from one GPU."""
        
        self.last_preemption_time = current_time
        
        print(f"-> [DEBUG] PREEMPT START: Job {self.id} being preempted from GPU {gpu_to_release.gpu_id}.")
        print(f"   [DEBUG] GPU state BEFORE release: Mem={gpu_to_release.available_memory:.2f}, Util={gpu_to_release.available_utilization:.2f}, Tasks={gpu_to_release.running_tasks.keys()}")
        
        gpu_to_release.release_task(self)
        
        print(f"   [DEBUG] GPU state AFTER release: Mem={gpu_to_release.available_memory:.2f}, Util={gpu_to_release.available_utilization:.2f}, Tasks={gpu_to_release.running_tasks.keys()}")
        
        if gpu_to_release in self.assigned_gpus:
            self.assigned_gpus.remove(gpu_to_release)
        
        # Re-distribute load over remaining GPUs
        for gpu in self.assigned_gpus:
            gpu.release_task(self)

        if self.assigned_gpus:
            self._distribute_load()

        self.paused_until = current_time + PREEMPTION_OVERHEAD
        print(f"<- [DEBUG] PREEMPT END: Job {self.id} is now running on {len(self.assigned_gpus)} GPUs.")
        
    def reclaim_gpu(self, gpu_to_add, current_time):
        if gpu_to_add in self.assigned_gpus: return

        for gpu in self.assigned_gpus:
            gpu.release_task(self)

        self.assigned_gpus.append(gpu_to_add)
        self._distribute_load()
        
        self.paused_until = current_time + RECLAMATION_OVERHEAD
    
    def update_progress(self, time_delta, current_time):
        """Updates remaining work using a normalized speedup factor."""
        if not self.assigned_gpus or current_time < self.paused_until:
            return
        
        if self.gpus_needed > 0:
            speedup_factor = len(self.assigned_gpus) / self.gpus_needed
        else:
            speedup_factor = 1 # Fallback

        self.remaining_work -= (time_delta * speedup_factor)
    
    def is_complete(self):
        return self.remaining_work <= 0

    def record_completion(self, current_time):
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time

    # --- NEW: Method to check preemption threshold ---
    def can_be_preempted(self, current_time, estimated_borrow_time=1000.0):
        """
        Predicts if a *future* preemption will violate the max end-time
        by checking past delays and adding future predicted delays.
        """
        if self.start_time == -1:
            return False 
        
        current_num_gpus = len(self.assigned_gpus)
        
        # Can't preempt if the job is already down to its last GPU
        if current_num_gpus <= 1:
            return False
            
        # 1. Calculate delay *already incurred* based on simulation state.
        work_done_in_ideal_time = self.ideal_duration - self.remaining_work
        time_elapsed = current_time - self.start_time
        current_delay_incurred = max(0, time_elapsed - work_done_in_ideal_time)

        # 2. Calculate *future* fixed overheads from this new preemption
        future_fixed_delay = PREEMPTION_OVERHEAD + RECLAMATION_OVERHEAD

        # 3. Calculate *future* "Slowdown Penalty" based on *current* GPU count
        
        current_speed_factor = current_num_gpus / self.gpus_needed
        new_speed_factor = (current_num_gpus - 1) / self.gpus_needed

        time_to_finish_new_slow = self.remaining_work / new_speed_factor
        
        if time_to_finish_new_slow <= estimated_borrow_time:
            # Case A: Job finishes *while* slow.
            time_to_finish_current = self.remaining_work / current_speed_factor
            slowdown_delay = time_to_finish_new_slow - time_to_finish_current
        
        else:
            # Case B: Job outlasts the borrow time.
            # Simplified formula for lost time: estimated_borrow_time / current_num_gpus
            slowdown_delay = estimated_borrow_time / current_num_gpus

        # 4. Calculate total *predicted* delay
        total_predicted_delay = (current_delay_incurred + 
                                 future_fixed_delay + 
                                 slowdown_delay)
        
        # 5. Calculate max *allowable* delay
        max_allowable_delay = self.max_allowable_duration - self.ideal_duration

        return total_predicted_delay <= max_allowable_delay