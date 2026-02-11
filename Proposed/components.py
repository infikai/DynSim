# file: components.py

GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
PREEMPTION_OVERHEAD = 3
RECLAMATION_OVERHEAD = 5
PREEMPTION_COOLDOWN = 5

# LLM Inference Performance Model Constants
LLM_BASE_TTFT = 2.5          
LLM_TKN_PER_INPUT = 0.005    
LLM_TPOT = 0.1               
LLM_MAX_CONCURRENCY = 16      

class SimulationClock:
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
        self.drain_at_time = -1 
        self.running_tasks = {}
        self.reclamation_cooldown_timer = 0

    def assign_task(self, job, mem_slice, util_slice):
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
        if not self.is_llm_server or self.llm_slots_available <= 0:
            return False
        if self.drain_at_time != -1 and current_time >= self.drain_at_time:
            return False 
        return True 

    def assign_llm_task(self, job):
        self.llm_slots_available -= 1
        self.running_tasks[job.id] = {'job': job, 'type': 'llm_inference'}

    def release_task(self, job):
        if job.id not in self.running_tasks: return
        task_info = self.running_tasks.pop(job.id)
        if task_info.get('type') == 'llm_inference':
            self.llm_slots_available += 1
        else: 
            self.available_memory += task_info['mem']
            self.available_utilization += task_info['util']
    
    def is_idle(self):
        return not self.running_tasks and not self.is_llm_server

    def get_running_training_jobs(self):
        return [task['job'] for task in self.running_tasks.values() if task['job'].job_type == 'training']

class Job:
    def __init__(self, id, job_type, arrival_time, base_duration=0, 
                 memory_required=0, utilization_required=0, 
                 input_tokens=0, output_tokens=0):
        self.id = id
        self.job_type = job_type
        self.arrival_time = arrival_time
        self.memory_required = max(memory_required, 1)
        self.utilization_required = max(utilization_required, 1)
        
        if self.job_type == 'llm_inference':
            self.base_duration = LLM_BASE_TTFT + (LLM_TKN_PER_INPUT * input_tokens) + (LLM_TPOT * output_tokens)
        else:
            self.base_duration = base_duration
        
        self.remaining_work = self.base_duration
        self.assigned_gpus = []
        self.start_time = -1
        self.completion_time = -1
        self.turnaround_time = -1
        self.paused_until = -1
        self.gpus_needed = 1
        self.last_preemption_time = -1
        self.ideal_duration = self.base_duration
        self.is_checkpointed = False
        
        # Training jobs can be delayed indefinitely when preempted by inference
        if self.job_type == 'training':
            self.max_allowable_duration = float('inf')
        else:
            self.max_allowable_duration = float('inf') 

    def assign_resources(self, gpus, current_time):
        self.assigned_gpus = gpus
        self.start_time = current_time
        self._distribute_load()

    def _distribute_load(self):
        if not self.assigned_gpus: return
        mem_per_gpu = self.memory_required / self.gpus_needed
        util_per_gpu = self.utilization_required / self.gpus_needed
        for gpu in self.assigned_gpus:
            gpu.assign_task(self, mem_per_gpu, util_per_gpu)

    def preempt_and_pause(self, gpu_to_release, current_time):
        self.last_preemption_time = current_time
        gpu_to_release.release_task(self)
        if gpu_to_release in self.assigned_gpus:
            self.assigned_gpus.remove(gpu_to_release)
        for gpu in self.assigned_gpus:
            gpu.release_task(self)
        if self.assigned_gpus:
            self._distribute_load()
        self.paused_until = current_time + PREEMPTION_OVERHEAD - 2
        
    def reclaim_gpu(self, gpu_to_add, current_time):
        if gpu_to_add in self.assigned_gpus: return
        for gpu in self.assigned_gpus:
            gpu.release_task(self)
        self.assigned_gpus.append(gpu_to_add)
        self._distribute_load()
        self.paused_until = current_time + RECLAMATION_OVERHEAD - 4

    def checkpoint_and_pause(self, gpu_to_release, current_time):
        """Checkpoint a 1-GPU job: release the GPU, preserve remaining_work."""
        self.last_preemption_time = current_time
        gpu_to_release.release_task(self)
        if gpu_to_release in self.assigned_gpus:
            self.assigned_gpus.remove(gpu_to_release)
        self.paused_until = current_time + PREEMPTION_OVERHEAD
        self.is_checkpointed = True

    def resume_from_checkpoint(self, gpu, current_time):
        """Resume a checkpointed job on a new GPU with no extra overhead."""
        self.assigned_gpus = [gpu]
        self.is_checkpointed = False
        self._distribute_load()
    
    def update_progress(self, time_delta, current_time):
        if not self.assigned_gpus or current_time < self.paused_until:
            return
        speedup_factor = len(self.assigned_gpus) / self.gpus_needed
        self.remaining_work -= (time_delta * speedup_factor)

    def can_be_preempted(self, current_time, estimated_borrow_time=1000.0):
        if self.start_time == -1:
            return False
        
        work_done = self.ideal_duration - self.remaining_work
        current_delay = max(0, (current_time - self.start_time) - work_done)
        future_fixed_delay = PREEMPTION_OVERHEAD + RECLAMATION_OVERHEAD - 5
        slowdown_delay = estimated_borrow_time / len(self.assigned_gpus)

        total_predicted_delay = current_delay + future_fixed_delay + slowdown_delay
        return total_predicted_delay <= (self.max_allowable_duration - self.ideal_duration)

    def is_complete(self):
        return self.remaining_work <= 0

    def record_completion(self, current_time):
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time