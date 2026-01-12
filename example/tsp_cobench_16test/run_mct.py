import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench_16test import TSPEvaluationCB
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.mcts_ahd.mcts_ahd import MCTS_AHD, MAProfiler
from llm4ad.method.mcts_ahd.resume import resume_ma
from llm4ad.tools.profiler import ProfilerBase


def main(
        host, key, model,
        max_sample_nums=100,
        max_api_cost=4.0,
        pop_size=20,
        init_size=4,
        selection_num=2,
        num_samplers=1,
        num_evaluators=1,
        alpha=0.5,
        lambda_0=0.1,
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

    method = MCTS_AHD(
        llm=llm, 
        profiler=MAProfiler(log_dir='logs_multitimes2/mcts_ahd/20260108_134903', log_style='complex', create_random_path=False),
        evaluation=task,
        max_sample_nums=max_sample_nums,
        max_api_cost=max_api_cost,
        pop_size=pop_size,
        init_size=init_size,
        selection_num=selection_num,
        num_samplers=num_samplers,
        num_evaluators=num_evaluators,
        alpha=alpha,
        lambda_0=lambda_0,
        debug_mode=True,
    )
    resume_ma(method, path='logs_multitimes2/mcts_ahd/20260108_134903')

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
        init_size=4,              # Initial population size
        selection_num=2,          # Number of selected individuals for crossover
        num_samplers=20,          # Parallel samplers
        num_evaluators=20,        # Parallel evaluators
        alpha=0.5,                # UCT formula parameter for exploration/exploitation balance
        lambda_0=0.1,             # UCT formula parameter for exploration/exploitation balance
    )
