# file: cluster_manager.py

from components import GPU, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN

class ClusterManager:
    """Manages two distinct GPU pools: Training and Inference."""
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_pool = [GPU(f"TrainGPU_{i}", pool_type='training') for i in range(num_training_gpus)]
        self.inference_pool = [GPU(f"InferGPU_{i}", pool_type='inference') for i in range(num_inference_gpus)]
        self.all_gpus = self.training_pool + self.inference_pool
        
        self.target_llm_servers = num_inference_gpus 
        print(f"ClusterManager initialized: {num_training_gpus} Training GPUs, {num_inference_gpus} Inference GPUs.")

    def pre_warm_llm_servers(self):
        servers_created = 0
        for gpu in self.inference_pool:
            gpu.convert_to_llm_server()
            servers_created += 1
        print(f"➡️ Successfully pre-warmed {servers_created} LLM servers in the Inference Pool.")

    def find_training_resources(self, count):
        """
        Standard finder: Idle Training GPUs + Empty Borrowed GPUs + Idle Inference GPUs.
        Used for simple checks, but dispatch logic now handles the complex reclamation.
        """
        train_pool_available = []
        for g in self.training_pool:
            if g.is_idle():
                train_pool_available.append(g)
            elif g.is_llm_server and not g.running_tasks:
                train_pool_available.append(g)

        if len(train_pool_available) >= count:
            return train_pool_available[:count]
        
        needed = count - len(train_pool_available)
        borrowable_gpus = [g for g in self.inference_pool if g.is_idle()]
        
        if len(borrowable_gpus) >= needed:
            return train_pool_available + borrowable_gpus[:needed]
            
        return [] 

    # --- NEW: Find Active 'Squatters' in Training Pool ---
    def find_non_idle_training_gpus_for_reclamation(self):
        """
        Returns Training Pool GPUs that are currently acting as active LLM servers.
        These are candidates to be 'drained' and returned to Training jobs.
        """
        return [g for g in self.training_pool if g.is_llm_server and g.running_tasks]

    def find_idle_resources_in_inference_pool(self):
        """
        Returns GPUs in the Inference Pool that are effectively idle 
        (either strictly idle or empty LLM servers).
        """
        return [g for g in self.inference_pool if g.is_idle() or (g.is_llm_server and not g.running_tasks)]

    def find_gpu_for_llm_job(self, current_time):
        for gpu in self.inference_pool:
            if gpu.is_available_for_llm(current_time):
                return gpu
        return None

    def find_idle_gpus_in_training_pool(self):
        return [g for g in self.training_pool if g.is_idle()]

    def find_idle_gpus_in_inference_pool(self):
        return [g for g in self.inference_pool if g.is_idle()]

    def find_borrowed_gpu_to_reclaim(self, current_time):
        potential_victims = []
        for gpu in self.inference_pool:
            if not gpu.is_llm_server and gpu.running_tasks:
                training_jobs = gpu.get_running_training_jobs()
                for job in training_jobs:
                    if current_time > job.last_preemption_time + PREEMPTION_COOLDOWN:
                         if job.can_be_preempted(current_time, estimated_borrow_time=1000.0):
                             potential_victims.append((job, gpu))

        if not potential_victims:
            return (None, None)
            
        potential_victims.sort(key=lambda item: item[0].last_preemption_time)
        return potential_victims[0]

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)

    def find_training_job_to_preempt_in_training_pool(self, current_time):
        """
        Finds a Training job running in the TRAINING pool to preempt (P5).
        """
        potential_victims = []
        for gpu in self.training_pool:
            # We look for GPUs running training jobs (not LLM servers, not idle)
            if not gpu.is_llm_server and gpu.running_tasks:
                training_jobs = gpu.get_running_training_jobs()
                for job in training_jobs:
                    if current_time > job.last_preemption_time + PREEMPTION_COOLDOWN:
                        if job.can_be_preempted(current_time, estimated_borrow_time=1000.0):
                            potential_victims.append((job, gpu))

        if not potential_victims:
            return (None, None)
        
        potential_victims.sort(key=lambda item: item[0].last_preemption_time)
        return potential_victims[0]