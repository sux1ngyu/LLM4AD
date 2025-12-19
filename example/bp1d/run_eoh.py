import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.bp_1d_construct import BP1DEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import ProfilerBase


def main(
        host,key,model,
        max_sample_nums=20000,
        max_generations=10,
        pop_size=4,
):

    llm = OpenAIAPI(base_url=host,  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   api_key=key,  # your key, e.g., 'sk-abcdefghijklmn'
                   model=model,  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = BP1DEvaluation()

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs/eoh', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=max_sample_nums,
                 max_generations=max_generations,
                 pop_size=pop_size,
                 num_samplers=1,
                 num_evaluators=1,
                 debug_mode=False)

    method.run()


if __name__ == '__main__':
    host = 'https://api.openai.com/v1'
    key = ''
    model = 'o3'
    main(
        host,key,model,
        max_sample_nums=100,
        max_generations=10,
        pop_size=4,
    )
