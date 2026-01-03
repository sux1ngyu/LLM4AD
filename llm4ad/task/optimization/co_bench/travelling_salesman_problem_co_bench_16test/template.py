template_program = '''
import numpy as np
import scipy.optimize as opt
import math
import random
from typing import List, Tuple, Dict
def solve(distance_matrix: np.ndarray, start_node: int = 0) -> dict:
    """
    Solve a TSP instance.
    Args:
        - distance_matrix (np.ndarray): n×n symmetric distance matrix where 
                                        distance_matrix[i][j] is the distance from node i to node j
                                        Format: 2D numpy array of shape (n, n)
        - start_node (int): Starting node index (default: 0)
    Returns:
        dict: Solution information with:
            - 'tour' (list): List of node indices representing the solution path
                            Format: [0, 3, 1, ...] where numbers are indices into the distance matrix
    """

    return {
        'tour': [],
    }
'''

task_description = ("The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where, "
                    "given a set of cities with known pairwise distances, the objective is to find the shortest "
                    "possible tour that visits each city exactly once and returns to the starting city. More "
                    "formally, given a complete graph G = (V, E) with vertices V representing cities and edges E with "
                    "weights representing distances, we seek to find a Hamiltonian cycle (a closed path visiting each "
                    "vertex exactly once) of minimum total weight."
                    "Help me design a novel algorithm to solve this problem.")
