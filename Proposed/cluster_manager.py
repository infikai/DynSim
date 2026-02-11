# file: cluster_manager.py
from components import GPU, PREEMPTION_COOLDOWN

class ClusterManager:
    def __init__(self, total_gpus):
        self.unified_pool = [GPU(f"GPU_{i}") for i in range(total_gpus)]
        
    def pre_warm_llm_servers(self, num_servers=500):
        # Pre-warm first N GPUs as LLM servers
        for i, gpu in enumerate(self.unified_pool):
            if i >= num_servers:
                break
            gpu.convert_to_llm_server()

    def find_gpu_for_llm_job(self, current_time):
        for gpu in self.unified_pool:
            if gpu.is_available_for_llm(current_time):
                return gpu
        return None

    def find_idle_gpus(self):
        return [g for g in self.unified_pool if g.is_idle()]

    def find_idle_or_convertible_gpus(self):
        return [g for g in self.unified_pool if g.is_idle() or (g.is_llm_server and not g.running_tasks)]

    def find_training_job_to_preempt(self, current_time):
        """Find a training job that can be preempted for inference."""
        potential = []
        for gpu in self.unified_pool:
            if not gpu.is_llm_server and gpu.running_tasks:
                for job in gpu.get_running_training_jobs():
                    if current_time > job.last_preemption_time + PREEMPTION_COOLDOWN:
                        if job.can_be_preempted(current_time):
                            potential.append((job, gpu))
        if not potential: 
            return None, None
        potential.sort(key=lambda x: x[0].last_preemption_time)
        return potential[0]

    def release_resources_for_job(self, job):
        for gpu in list(job.assigned_gpus):
            gpu.release_task(job)