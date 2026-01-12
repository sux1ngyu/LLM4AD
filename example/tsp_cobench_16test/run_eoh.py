### Test Only ###
# Set system path

import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench_16test import TSPEvaluationCB
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.eoh.resume import resume_eoh


def main(
        host,key,model,
        max_sample_nums=20000,
        max_generations=10,
        max_api_cost=4.0,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
):

    llm = OpenAIAPI(base_url=host,  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   api_key=key,  # your key, e.g., 'sk-abcdefghijklmn'
                   model=model,  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = TSPEvaluationCB(timeout_seconds=1000000,
                           split="dev", 
                           instance_timeout_seconds=1, 
                           data_dir='/home/wendiyu/projects/MultiReason/examples/tsp_optimization/tsplib_train/',
                           init_tester_path='matrix_init_random_coor') #timeout controlled by instance 10s

    method = EoH(llm=llm,
                 profiler=EoHProfiler(log_dir='logs_multitimes2/eoh/20260108_061549', log_style='complex', create_random_path=False),
                 evaluation=task,
                 max_sample_nums=max_sample_nums,
                 max_generations=max_generations,
                 max_api_cost=max_api_cost,
                 pop_size=pop_size,
                 num_samplers=num_samplers,
                 num_evaluators=num_evaluators,
                 debug_mode=True
                 )
    resume_eoh(method, path='logs_multitimes2/eoh/20260108_061549')
    method.run()


if __name__ == '__main__':
    host = 'https://api.openai.com/v1'
    key = ''
    model = 'o3'
    main(
        host,key,model,
        max_sample_nums=15000,
        max_generations=10000,
        max_api_cost=20.0,       #real limit is api cost, not generations or sample nums
        pop_size=20,
        num_samplers=20,
        num_evaluators=20,
    )
