import sys

sys.path.append('../../')  # This is for finding all the modules
from llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench_16test import TSPEvaluationCB
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.funsearch import FunSearch, FunSearchProfiler
from llm4ad.method.funsearch.resume import resume_funsearch



def main(
        host,key,model,
        max_sample_nums=20000,
        num_samplers=1,
        num_evaluators=1,
        max_api_cost=4.0,
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


    method = FunSearch(
        llm=llm, 
        profiler=FunSearchProfiler(log_dir='logs_multitimes2/funsearch/20260108_104825', log_style='complex', create_random_path=False),
        evaluation=task,
        max_sample_nums=max_sample_nums,
        num_samplers=num_samplers,
        num_evaluators=num_evaluators,
        max_api_cost=max_api_cost,
        debug_mode=True,
    )
    resume_funsearch(method, path='logs_multitimes2/funsearch/20260108_104825')
    method.run()


if __name__ == '__main__':
    host = 'https://api.openai.com/v1'
    key = ''
    model = 'o3'
    main(
        host,key,model,
        max_sample_nums=100000,#controlled by cost
        num_samplers=20,
        num_evaluators=20,
        max_api_cost=20.0,
    )
