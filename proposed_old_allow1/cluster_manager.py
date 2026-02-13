# file: cluster_manager.py

from components import GPU, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN

class ClusterManager:
    """Manages a single, unified pool of GPUs."""
    def __init__(self, num_gpus, target_llm_servers):
        # --- MODIFIED: Create a single, unified GPU pool ---
        self.gpus = [GPU(f"GPU_{i}") for i in range(num_gpus)]
        self.target_llm_servers = target_llm_servers
        print(f"ClusterManager initialized with {num_gpus} total GPUs.")

    # --- REMOVED: pre_warm_llm_servers method ---

    def find_gpu_for_stackable_inference(self, job):
        """Finds a single non-LLM GPU that can fit a small inference job."""
        # --- MODIFIED: Search the unified pool ---
        for gpu in sorted(self.gpus, key=lambda g: g.is_idle()):
            # Can't stack on an active LLM server
            if not gpu.is_llm_server and gpu.can_fit(job):
                return gpu
        return None

    def find_gpu_for_llm_job(self, current_time): # <-- MODIFIED
        """
        Finds a single existing LLM server GPU with available slots
        that is not currently in its "draining" phase.
        
        Priority:
        1. Find an existing LLM server with available slots.
        """
        for gpu in self.gpus:
            # --- MODIFIED: Use new helper which checks drain time ---
            if gpu.is_available_for_llm(current_time):
                return gpu
        return None # Return None if no existing server has slots
    
    def find_idle_gpus(self, count):
        """Finds a specific number of idle GPUs from the entire pool."""
        # --- MODIFIED: Search the unified pool ---
        # An idle GPU is one that is NOT an LLM server and has no tasks
        idle_gpus = [gpu for gpu in self.gpus if gpu.is_idle()]
        if len(idle_gpus) >= count:
            return idle_gpus[:count]
        return []

    def find_preemptible_job(self, current_time):
        """
        Finds the best training job to preempt based on:
        1. Not in cooldown.
        2. Estimated completion time stays within the threshold.
        3. Least Recently Preempted.
        """
        potential_victims = []
        
        for gpu in self.gpus:
            if not gpu.is_llm_server:
                for job in gpu.get_running_training_jobs():
                    if current_time > job.last_preemption_time + PREEMPTION_COOLDOWN or current_time == job.last_preemption_time:
                        
                        # --- MODIFIED: Use estimated_borrow_time=1000.0 ---
                        if job.can_be_preempted(current_time, estimated_borrow_time=1000.0):
                            potential_victims.append((job, gpu))
                    
        if not potential_victims:
            return (None, None)
            
        potential_victims.sort(key=lambda item: item[0].last_preemption_time)
        
        return potential_victims[0] # Return the best victim

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)