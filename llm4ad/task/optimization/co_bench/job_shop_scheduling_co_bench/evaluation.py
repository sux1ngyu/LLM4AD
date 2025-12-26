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
from llm4ad.task.optimization.co_bench.utils import load_subdir_as_text, select_indices_by_split
from llm4ad.task.optimization.co_bench.job_shop_scheduling_co_bench.template import template_program, task_description

__all__ = ['JSSEvaluationCB']


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


def _run_solve_worker(eva: Any, args: tuple[Any, ...], out_q: "mp.Queue") -> None:
    try:
        if not callable(eva):
            raise ValueError("Evaluator received a non-callable solve().")
        out_q.put(eva(*args))
    except Exception as e:
        out_q.put(f"Exception: {e}")


def _solve_with_timeout(eva: Any, args: tuple[Any, ...], timeout_seconds: float) -> Any:
    out_q: "mp.Queue" = mp.Queue()
    p = mp.Process(target=_run_solve_worker, args=(eva, args, out_q))
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


class JSSEvaluationCB(Evaluation):

    def __init__(self,
                 timeout_seconds=50,
                 instance_timeout_seconds: float = 10,
                 split: Literal["all", "dev", "test"] = "all",
                 **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.split: Literal["all", "dev", "test"] = split
        self.instance_timeout_seconds = float(instance_timeout_seconds)

        # Load datasets from Hugging Face
        dataset = load_subdir_as_text("CO-Bench/CO-Bench", "Job shop scheduling")
        self._datasets = {}
        for filename in dataset:
            # Join all text rows into a single string
            text_content = '\n'.join([row['text'] for row in dataset[filename]])
            self._datasets[filename] = text_content

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        split = kwargs.get("split", None)
        return self.evaluate(callable_func, program_str=program_str, split=split)

    def evaluate(
        self,
        eva: callable,
        *,
        program_str: str | None = None,
        split: Literal["all", "dev", "test"] | None = None,
    ) -> float | None:
        split_to_use: Literal["all", "dev", "test"] = (split or self.split)
        dev_map = self.get_dev()

        case_scores: list[float] = []
        per_case: dict[str, list[dict[str, Any]]] = {}

        for filename, ins_text in self._datasets.items():
            cases = self.load_data(ins_text)

            if dev_map is not None and split_to_use == "dev" and filename not in dev_map:
                continue

            dev_indices = None if dev_map is None else dev_map.get(filename, None)
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
                    (j["n_jobs"], j["n_machines"], j["times"], j["machines"]),
                    self.instance_timeout_seconds,
                )
                if isinstance(solve_res, str):
                    scores_full[idx] = solve_res
                    continue
                try:
                    scores_full[idx] = self.eval_func(
                        j["n_jobs"],
                        j["n_machines"],
                        j["times"],
                        j["machines"],
                        solve_res["start_times"],
                        lower_bound=j["lower_bound"],
                        upper_bound=j["upper_bound"],
                    )
                except Exception as e:
                    scores_full[idx] = f"Exception: {e}"

            per_case[filename] = []
            vals = [float(scores_full[i]) if isinstance(scores_full[i], (int, float)) else 0.0 for i in selected_indices]
            for i in selected_indices:
                raw = scores_full[i]
                if isinstance(raw, str) and raw.startswith("Timeout"):
                    status = "timeout"
                elif isinstance(raw, str) and raw.startswith("Exception"):
                    status = "exception"
                elif isinstance(raw, (int, float)):
                    status = "ok"
                else:
                    status = "other"
                per_case[filename].append({"idx": int(i), "status": status, "raw": raw, "normed": raw})
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

    def load_data(self, input_string):
        cases = []
        lines = [line.strip() for line in input_string.split('\n') if line.strip()]  # remove blank lines

        i = 0
        while i < len(lines):
            # Look for a header line starting with "Nb of jobs"
            if lines[i].startswith("Nb of jobs"):
                # Next line contains six numbers: n_jobs, n_machines, time_seed, machine_seed, upper_bound, lower_bound
                i += 1
                header_tokens = lines[i].split()
                if len(header_tokens) < 6:
                    raise ValueError("Header line does not contain 6 values.")
                n_jobs = int(header_tokens[0])
                n_machines = int(header_tokens[1])
                time_seed = int(header_tokens[2])
                machine_seed = int(header_tokens[3])
                upper_bound = int(header_tokens[4])
                lower_bound = int(header_tokens[5])

                # Find the "Times" section
                i += 1
                if not lines[i].lower().startswith("times"):
                    raise ValueError("Expected 'Times' section, got: " + lines[i])
                i += 1  # move to first line of times
                times = []
                for _ in range(n_jobs):
                    # Each line should contain n_machines numbers
                    time_line = list(map(int, lines[i].split()))
                    if len(time_line) != n_machines:
                        raise ValueError(f"Expected {n_machines} numbers in times row, got {len(time_line)}")
                    times.append(time_line)
                    i += 1

                # Find the "Machines" section
                if i >= len(lines) or not lines[i].lower().startswith("machines"):
                    raise ValueError("Expected 'Machines' section, got: " + (lines[i] if i < len(lines) else "EOF"))
                i += 1  # move to first line of machines
                machines = []
                for _ in range(n_jobs):
                    machine_line = list(map(int, lines[i].split()))
                    if len(machine_line) != n_machines:
                        raise ValueError(f"Expected {n_machines} numbers in machines row, got {len(machine_line)}")
                    machines.append(machine_line)
                    i += 1

                # Build the test case dictionary and add to the list of cases.
                case = {
                    "n_jobs": n_jobs,
                    "n_machines": n_machines,
                    "time_seed": time_seed,
                    "machine_seed": machine_seed,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                    "times": times,
                    "machines": machines
                }
                cases.append(case)
            else:
                # If the current line is not a header, skip it.
                i += 1

        return cases

    def eval_func(self, n_jobs, n_machines, times, machines, start_times, **kwargs):
        """
        Evaluates the solution for a job shop scheduling problem.
        Input:
            n_jobs (int): Number of jobs.
            n_machines (int): Number of machines.
            times (list of list of int): Processing times for each operation.
                Dimensions: n_jobs x n_machines.
            machines (list of list of int): Machine assignments for each operation.
                Dimensions: n_jobs x n_machines.
            start_times (list of list of int): Proposed start times for each operation.
                Dimensions: n_jobs x n_machines.
            kwargs: Other parameters that may be provided, which are ignored here.
        Output:
            score (int): The makespan, defined as the maximum completion time across all jobs.
        Raises:
            ValueError: If any scheduling constraints are violated.
        """

        # Check that start_times dimensions match the problem dimensions.
        if len(start_times) != n_jobs:
            raise ValueError(f"Expected start_times to have {n_jobs} rows, got {len(start_times)}")
        for i, row in enumerate(start_times):
            if len(row) != n_machines:
                raise ValueError(f"Expected start_times row {i} to have {n_machines} entries, got {len(row)}")
            for t in row:
                if t < 0:
                    raise ValueError("Start times must be non-negative.")

        # Constraint (i): Sequential processing for each job.
        job_completion_times = []
        for i in range(n_jobs):
            current_time = None
            for j in range(n_machines):
                st = start_times[i][j]
                pt = times[i][j]
                if j == 0:
                    # For the first operation, simply set the finish time.
                    current_time = st + pt
                else:
                    # For subsequent operations, the start time must be no earlier than the finish of the previous.
                    if st < current_time:
                        raise ValueError(
                            f"Job {i} operation {j} starts at {st} but previous operation finishes at {current_time}")
                    current_time = st + pt
            job_completion_times.append(current_time)

        # Constraint (ii): Machine non-overlap.
        # Build a dictionary mapping machine id to a list of (start_time, finish_time, job, op_index)
        machine_schedules = {}
        for i in range(n_jobs):
            for j in range(n_machines):
                machine_id = machines[i][j]
                st = start_times[i][j]
                pt = times[i][j]
                finish_time = st + pt
                if machine_id not in machine_schedules:
                    machine_schedules[machine_id] = []
                machine_schedules[machine_id].append((st, finish_time, i, j))

        # For each machine, sort operations by start time and check for overlaps.
        for machine_id, ops in machine_schedules.items():
            ops_sorted = sorted(ops, key=lambda x: x[0])
            for k in range(1, len(ops_sorted)):
                prev_st, prev_finish, prev_job, prev_op = ops_sorted[k - 1]
                curr_st, curr_finish, curr_job, curr_op = ops_sorted[k]
                if prev_finish > curr_st:
                    raise ValueError(
                        f"Machine {machine_id}: Operation from job {prev_job}, op {prev_op} (finishing at {prev_finish}) overlaps with job {curr_job}, op {curr_op} (starting at {curr_st}).")

        # Compute the makespan as the maximum completion time among all jobs.
        makespan = max(job_completion_times)

        score = kwargs['lower_bound'] / makespan

        return score

    def get_dev(self):
        dev = {'tai100_20.txt': [1, 8, 0, 6, 9], 'tai15_15.txt': [1, 8, 9, 4, 5], 'tai20_15.txt': [2, 7, 0, 8, 3],
               'tai20_20.txt': [9, 7, 8, 3, 0], 'tai30_15.txt': [8, 7, 2, 5, 1], 'tai30_20.txt': [0, 5, 1, 4, 6],
               'tai50_15.txt': [9, 1, 4, 5, 6], 'tai50_20.txt': [5, 9, 7, 4, 8]}

        return dev




