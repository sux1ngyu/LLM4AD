import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench_16test import TSPEvaluationCB
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
from llm4ad.tools.profiler import ProfilerBase


def main(
        host, key, model,
        max_sample_nums=100,
        max_api_cost=4.0,
        pop_size=20,
        mutation_rate=0.5,
        num_samplers=1,
        num_evaluators=1,
):
    llm = OpenAIAPI(
        base_url=host,  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
        api_key=key,    # your key, e.g., 'sk-abcdefghijklmn'
        model=model,    # your llm, e.g., 'gpt-3.5-turbo', 'o3'
        timeout=60
    )

    task = TSPEvaluationCB(timeout_seconds=1000000,
                       split="dev", 
                       instance_timeout_seconds=1, 
                       data_dir='/home/wendiyu/projects/MultiReason/examples/tsp_optimization/tsplib_train/',
                       init_tester_path='matrix_init_random_coor') #timeout controlled by instance 10s

    method = ReEvo(
        llm=llm, 
        profiler=ReEvoProfiler(log_dir='logs_multitimes2/reevo/', log_style='complex'),
        evaluation=task,
        max_sample_nums=max_sample_nums,
        max_api_cost=max_api_cost,
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        num_samplers=num_samplers,
        num_evaluators=num_evaluators,
        debug_mode=True,
    )

    method.run()


if __name__ == '__main__':
    host = 'https://api.openai.com/v1'
    key = ''
    model = 'o3'
    main(
        host, key, model,
        max_sample_nums=100000,   # Total samples to generate
        max_api_cost=20.0,         # API budget in USD
        pop_size=20,              # Population size
        mutation_rate=0.5,        # Mutation probability
        num_samplers=20,           # Parallel samplers
        num_evaluators=20,         # Parallel evaluators
    )
