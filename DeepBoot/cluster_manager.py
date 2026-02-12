# file: cluster_manager.py

from components import GPU, LLM_MAX_CONCURRENCY

class ClusterManager:
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        print(f"ClusterManager initialized: {num_training_gpus} Training GPUs, {num_inference_gpus} Inference GPUs.")

    def find_resources_for_llm_batch(self, num_jobs_needed):
        available_gpus = [] 
        
        active_server_gpus = [gpu for gpu in self.inference_gpus if gpu.is_llm_server]
        available_gpus.extend(active_server_gpus)

        convertible_gpus = [gpu for gpu in self.inference_gpus if not gpu.is_llm_server and gpu.is_idle()]
        available_gpus.extend(convertible_gpus)

        return available_gpus, []

    def find_gpu_for_llm_job(self):
        for gpu in self.inference_gpus:
            if gpu.is_llm_server and gpu.llm_slots_available > 0:
                return gpu
        
        for gpu in self.inference_gpus:
            if not gpu.is_llm_server and gpu.is_idle():
                return gpu
                
        return None

    def find_reclaim_target(self, can_preempt_fn=None):
        """
        Find a LOANED GPU (TRAIN state) to preempt, respecting deadline constraints.
        
        Args:
            can_preempt_fn: Optional callback(job, gpu) -> bool to check deadline constraints
        """
        best_gpu = None
        min_loss = float('inf')

        # Only look at GPUs currently loaned to training
        loaned_gpus = [g for g in self.inference_gpus if g.state == 'TRAIN']

        for gpu in loaned_gpus:
            # Skip initializing GPUs (they are in TRAIN but have no tasks yet)
            if not gpu.running_tasks: continue
            
            job = list(gpu.running_tasks.values())[0]['job']
            
            # NEW: Check deadline constraint
            if can_preempt_fn and not can_preempt_fn(job, gpu):
                continue  # Skip this GPU, preemption would violate deadline
            
            current_gpus = len(job.assigned_gpus)
            
            if current_gpus <= 1:
                # FIX: Assign a high (but finite) loss to allow preemption if necessary
                # or strictly enforce preemption for LLM bursts.
                loss = 1000.0 
            else:
                curr_speed = job.calculate_speedup(current_gpus)
                next_speed = job.calculate_speedup(current_gpus - 1)
                loss = curr_speed - next_speed
            
            if loss < min_loss:
                min_loss = loss
                best_gpu = gpu
                
        return best_gpu



    def find_idle_gpus_for_training(self, count):
        idle_gpus = [gpu for gpu in self.training_gpus if gpu.is_idle()]
        return idle_gpus[:count] if len(idle_gpus) >= count else []

    def find_loanable_inference_gpus(self, count):
        loanable = []
        for gpu in self.inference_gpus:
            if gpu.state == 'FREE':
                loanable.append(gpu)
                if len(loanable) == count: break
        return loanable

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)