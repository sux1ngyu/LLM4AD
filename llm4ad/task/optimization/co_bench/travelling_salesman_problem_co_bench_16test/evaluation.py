# References:
#   - Sun, W., Feng, S., Li, S., & Yang, Y. Co-bench: Benchmarking language
#       model agents in algorithm search for combinatorial optimization.
#       arXiv preprint arXiv:2504.04310 (2025).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Literal
import multiprocessing as mp
import os
import json
import hashlib
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.co_bench.utils import select_indices_by_split
from llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench_matrix.template import template_program, task_description

__all__ = ['TSPEvaluationCB']


def _kill_process_tree(pid: int) -> None:
    try:
        import psutil  # type: ignore

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        try:
            parent.kill()
        except Exception:
            pass
    except Exception:
        pass


def _run_solve_worker(eva: Any, distance_matrix: Any, out_q: "mp.Queue") -> None:
    try:
        if not callable(eva):
            raise ValueError("Evaluator received a non-callable solve().")
        out_q.put(eva(distance_matrix))
    except Exception as e:
        out_q.put(f"Exception: {e}")


def _solve_with_timeout(
    eva: Any,
    distance_matrix: Any,
    timeout_seconds: float,
) -> Any:
    out_q: "mp.Queue" = mp.Queue()
    p = mp.Process(target=_run_solve_worker, args=(eva, distance_matrix, out_q))
    p.start()
    p.join(timeout_seconds + 1)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        _kill_process_tree(p.pid)
        try:
            p.join(1)
        except Exception:
            pass
        return f"Timeout ({timeout_seconds}s)"
    try:
        return out_q.get_nowait()
    except Exception:
        return "No result"


def _append_jsonl(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        import fcntl  # type: ignore

        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


class TSPEvaluationCB(Evaluation):

    def __init__(self,
                 timeout_seconds=50,
                 instance_timeout_seconds: float = 10,
                 split: Literal["all", "dev", "test"] = "all",
                 data_dir: str = None,
                 init_tester_path: str = None,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.split: Literal["all", "dev", "test"] = split
        self.instance_timeout_seconds = float(instance_timeout_seconds)
        
        if data_dir and init_tester_path:
            category_dir = os.path.join(data_dir, init_tester_path)
            instances = self._load_npz_instances(category_dir)
            self._datasets = {init_tester_path: instances}
        else:
            raise ValueError("data_dir and init_tester_path must be provided")

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        split = kwargs.get("split", None)
        return self.evaluate(callable_func, program_str=program_str, split=split)

    def evaluate(
        self,
        eva: callable,
        *,
        program_str: str | None = None,
        split: Literal["all", "dev", "test"] = None,
    ) -> float | None:
        split_to_use: Literal["all", "dev", "test"] = (split or self.split)
        dev_map = self.get_dev()

        case_scores: list[float] = []
        per_case: dict[str, list[dict[str, Any]]] = {}
        for category_name, instances_list in self._datasets.items():
            cases = instances_list

            if dev_map is not None and split_to_use == "dev" and category_name not in dev_map:
                continue

            dev_indices = None if dev_map is None else dev_map.get(category_name, None)
            selected_indices = select_indices_by_split(
                len(cases),
                split=split_to_use,
                dev_indices=dev_indices,
            )
            if not selected_indices:
                continue

            scores_full: list[Any] = ["Skipped"] * len(cases)
            for idx in selected_indices:
                j = cases[idx]
                solve_res = _solve_with_timeout(
                    eva,
                    j["distance_matrix"],
                    self.instance_timeout_seconds,
                )
                if isinstance(solve_res, str):
                    scores_full[idx] = solve_res
                    continue
                try:
                    scores_full[idx] = self.eval_func(
                        j["distance_matrix"],
                        None,
                        solve_res["tour"],
                    )
                except Exception as e:
                    scores_full[idx] = f"Exception: {e}"

            normed = self.norm_score({category_name: (scores_full, None)}, cases)
            if category_name not in normed:
                continue
            normed_scores_full, _ = normed[category_name]
            per_case[category_name] = []
            vals = [
                float(normed_scores_full[i]) if isinstance(normed_scores_full[i], (int, float)) else 0.0
                for i in selected_indices
            ]
            for i in selected_indices:
                raw = scores_full[i]
                normed_v = normed_scores_full[i]
                if isinstance(raw, str) and raw.startswith("Timeout"):
                    status = "timeout"
                elif isinstance(raw, str) and raw.startswith("Exception"):
                    status = "exception"
                elif isinstance(raw, (int, float)):
                    status = "ok"
                else:
                    status = "other"
                per_case[category_name].append(
                    {
                        "idx": int(i),
                        "status": status,
                        "raw": raw,
                        "normed": normed_v,
                    }
                )
            if vals:
                case_scores.append(float(np.mean(vals)))

        if not case_scores:
            return None
        overall = float(np.mean(case_scores))

        run_log_dir = os.environ.get("LLM4AD_RUN_LOG_DIR", "").strip()
        if run_log_dir:
            prog_hash = None
            if program_str:
                prog_hash = hashlib.sha1(program_str.encode("utf-8", errors="ignore")).hexdigest()[:12]
            _append_jsonl(
                os.path.join(run_log_dir, "instance_scores.jsonl"),
                {
                    "task": self.__class__.__name__,
                    "split": split_to_use,
                    "instance_timeout_seconds": self.instance_timeout_seconds,
                    "score": overall,
                    "program_sha1_12": prog_hash,
                    "cases": per_case,
                },
            )

        return overall

    def _load_npz_instances(self, directory: str) -> list:
        """Load TSP instances from .npz files in a directory."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' not found.")
        
        instances = []
        npz_files = sorted([f for f in os.listdir(directory) if f.endswith('.npz')])
        
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in '{directory}'.")
        
        for npz_file in npz_files:
            filepath = os.path.join(directory, npz_file)
            data = np.load(filepath)
            
            instance_name = npz_file.replace('.npz', '')
            instance_info = {
                'name': instance_name,
                'distance_matrix': data['distance_matrix'],
                'nn_bound': float(data['nn_bound']),
            }
            
            if 'lower_bound' in data:
                instance_info['lower_bound'] = float(data['lower_bound'])
            if 'upper_bound' in data:
                instance_info['upper_bound'] = float(data['upper_bound'])
            if 'generator_type' in data:
                instance_info['generator_type'] = data['generator_type'].item() if hasattr(data['generator_type'], 'item') else str(data['generator_type'])
            
            instances.append(instance_info)
        
        return instances

    def eval_func(self, distance_matrix, label_tour, tour):
        """
        Evaluate a predicted TSP tour against a reference tour.
        Args:
            distance_matrix (np.ndarray): n×n distance matrix
            label_tour (list): Reference/optimal tour as list of node indices
                              Format: [0, 3, 1, ...] (may be None if no reference available)
            tour (list): Predicted tour from the solver as list of node indices
                             Format: [0, 3, 1, ...]
        Returns:
            float: Predicted tour cost
        """
        num_nodes = distance_matrix.shape[0]

        if len(tour) != num_nodes:
            raise Exception(f"Invalid tour length: Expected {num_nodes}, got {len(tour)}")
        nodes_set = set(tour)

        if len(nodes_set) != num_nodes:
            raise Exception(f"Invalid tour: Contains {len(nodes_set)} unique nodes, expected {num_nodes}")

        expected_nodes = set(range(num_nodes))
        if nodes_set != expected_nodes:
            raise Exception(f"Invalid tour: Contains out-of-range or missing nodes")

        cost = 0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]
            cost += distance_matrix[from_node][to_node]

        return cost

    def norm_score(self, results, instances_list: list = None):
        normed = {}
        for case, (scores, error_message) in results.items():
            if instances_list is None:
                continue
            normed_scores = []
            for idx, score in enumerate(scores):
                if isinstance(score, (int, float)) and idx < len(instances_list):
                    nn_bound = instances_list[idx].get('nn_bound', 1.0)
                    if nn_bound > 0:
                        normed_scores.append(nn_bound / score)
                    else:
                        normed_scores.append(score)
                else:
                    normed_scores.append(score)
            normed[case] = (normed_scores, error_message)
        return normed

    def get_dev(self):
        # TSP does not have a dev split defined in CO-Bench
        return None







