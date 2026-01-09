from __future__ import annotations

import copy
import json
import os.path
import re

from tqdm.auto import tqdm

from .mcts_ahd import MCTS_AHD
from .profiler import MAProfiler
from .population import Population
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_latest_pop_json(log_path: str):
    path = os.path.join(log_path, 'population')
    orders = []
    for p in os.listdir(path):
        order = int(p.split('.')[0].split('_')[1])
        orders.append(order)
    max_o = max(orders)
    return os.path.join(path, f'pop_{max_o}.json'), max_o


def _get_all_samples_and_scores(path, get_algorithm=True):
    file_dir = os.path.join(path, 'samples')
    # get all file directories
    all_files = os.listdir(file_dir)
    # filer `samples_*.json` files and ignore `samples_best.json`
    sample_files = [f for f in all_files if f.startswith('samples_') and f != 'samples_best.json']

    def extract_number(filename):
        # match the first number of the filename
        match = re.search(r'samples_(\d+)~', filename)
        if match:
            return int(match.group(1))
        return 0

    sorted_files = sorted(sample_files, key=extract_number)

    all_func = []
    all_score = []
    all_algorithm = []
    max_o = 0  # the max sample orders
    last_total_cost = 0.0  # Track the last total_cost to restore LLM's total_api_cost

    for file in sorted_files:
        file_path = os.path.join(file_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            for sample in samples:
                func = sample['function']
                acc = sample['score'] if sample['score'] else float('-inf')
                all_func.append(func)
                all_score.append(acc)
                all_algorithm.append(sample['algorithm'])
                max_o = sample['sample_order']
                # Track the last total_cost value
                if 'total_cost' in sample and sample['total_cost'] is not None:
                    last_total_cost = max(last_total_cost, sample['total_cost'])

    if get_algorithm:
        return all_func, all_score, max_o, all_algorithm, last_total_cost
    return all_func, all_score, max_o, last_total_cost


# def _get_all_samples_and_scores(path):
#     path = os.path.join(path, 'samples')
#
#     def path_to_int(path):
#         num = int(path.split('.')[0].split('_')[1])
#         return num
#
#     all_func = []
#     all_score = []
#     dirs = list(os.listdir(path))
#     dirs = sorted(dirs, key=path_to_int)
#     max_o = path_to_int(dirs[-1])
#
#     for dir in dirs:
#         file_name = os.path.join(path, dir)
#         with open(file_name, 'r') as f:
#             sample = json.load(f)
#         func = sample['function']
#         acc = sample['score'] if sample['score'] else float('-inf')
#         all_func.append(func)
#         all_score.append(acc)
#
#     return all_func, all_score, max_o


def _resume_pop(log_path: str, init_pop_size, pop_size) -> Population:
    path, max_gen = _get_latest_pop_json(log_path)
    print(f'RESUME MCTS_AHD: Generations: {max_gen}.', flush=True)
    with open(path, 'r') as f:
        data = json.load(f)
    pop = Population(init_pop_size=init_pop_size, pop_size=pop_size)
    for d in data:
        func = d['function']
        func = tfpc.text_to_function(func)
        score = d['score']
        algorithm = d['algorithm']
        func.score = score
        func.algorithm = algorithm
        pop.register_function(func)
    pop._generation = max_gen
    return pop


def _resume_text2func(f, s, template_func: Function):
    temp = copy.deepcopy(template_func)
    f = tfpc.text_to_function(f)
    if f is None:
        temp.body = '    pass'
        temp.score = None
        return temp
    else:
        f.score = s
        return f


def _resume_pf(log_path: str, pf: MAProfiler, template_func):
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, sample_max_order, algorithms, last_total_cost = _get_all_samples_and_scores(log_path)
    print(f'RESUME MCTS_AHD: Sample order: {sample_max_order}.', flush=True)
    # Note: MAProfiler doesn't have _prog_db_order, so we don't need to set it
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume MCTS_AHD Profiler'):  # noqa
        f, s, algo = funcs[i], scores[i], algorithms[i]
        f = _resume_text2func(f, s, template_func)
        f.algorithm = algo
        pf.register_function(f, resume_mode=True)
    return last_total_cost


def resume_ma(ma: MCTS_AHD, path):
    ma._resume_mode = True
    pf = ma._profiler
    log_path = path
    # resume program database
    pop = _resume_pop(log_path, ma._init_pop_size, ma._pop_size)
    ma._population = pop
    # resume profiler and get last total_cost
    template_func = ma._function_to_evolve
    last_total_cost = _resume_pf(log_path, pf, template_func)
    # resume mcts_ahd
    _, _, sample_max_order, _, _ = _get_all_samples_and_scores(log_path)
    ma._tot_sample_nums = sample_max_order
    # Restore LLM's total_api_cost from the last sample's total_cost
    if last_total_cost > 0:
        ma._sampler.llm.total_api_cost = last_total_cost
        print(f'RESUME MCTS_AHD: Restored LLM total_api_cost to ${last_total_cost:.6f}', flush=True)
