# file: cluster_manager.py

from components import GPU

class ClusterManager:
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        print(f"ClusterManager initialized: {num_training_gpus} Training GPUs, {num_inference_gpus} Inference GPUs.")

    def find_reclaim_target(self, can_preempt_fn=None):
        """
        Find a LOANED GPU (TRAIN state) to preempt, respecting deadline constraints.
        
        Args:
            can_preempt_fn: Optional callback(job, gpu) -> bool to check deadline constraints
        
        Returns:
            (best_gpu, deadline_blocked_count) - best_gpu is None if no target found,
            deadline_blocked_count is how many candidates were skipped due to deadlines.
        """
        best_gpu = None
        min_loss = float('inf')
        deadline_blocked = 0

        # Only look at GPUs currently loaned to training
        loaned_gpus = [g for g in self.inference_gpus if g.state == 'TRAIN']

        for gpu in loaned_gpus:
            # Skip initializing GPUs (they are in TRAIN but have no tasks yet)
            if not gpu.running_tasks: continue
            
            job = list(gpu.running_tasks.values())[0]['job']
            
            # Check deadline constraint
            if can_preempt_fn and not can_preempt_fn(job, gpu):
                deadline_blocked += 1
                continue  # Skip this GPU, preemption would violate deadline
            
            current_gpus = len(job.assigned_gpus)
            
            if current_gpus <= 1:
                loss = 1000.0 
            else:
                curr_speed = job.calculate_speedup(current_gpus)
                next_speed = job.calculate_speedup(current_gpus - 1)
                loss = curr_speed - next_speed
            
            if loss < min_loss:
                min_loss = loss
                best_gpu = gpu
                
        return best_gpu, deadline_blocked

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)