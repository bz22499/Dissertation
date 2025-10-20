# initial_program.py
import itertools
import numpy as np
import sys

def solve(n: int) -> np.ndarray:
    """
    Returns a large cap set in `n` dimensions using a priority-based greedy approach.
    """
    all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    
    # Calculate priorities for each vector - THIS IS THE EVOLVABLE PART
    priorities = np.array([priority_function(tuple(vector), n) for vector in all_vectors], dtype=float)
    
    capset = np.empty(shape=(0, n), dtype=np.int32)
    
    while np.any(priorities != -np.inf):
        max_index = np.argmax(priorities)
        vector = all_vectors[None, max_index]  # Shape [1, n]
        
        # Block vectors that would create lines
        if capset.shape[0] > 0:
            blocking = np.einsum('cn,n->c', (-capset - vector) % 3, powers)
            priorities[blocking] = -np.inf
            
        priorities[max_index] = -np.inf
        capset = np.concatenate([capset, vector], axis=0)
    
    return capset

def priority_function(v: tuple, n: int) -> float:
    """
    Priority function to be evolved by OpenEvolve.
    This should be the main target for LLM evolution.
    """
    # Start with a simple baseline that can be improved
    return sum(x for x in v if x != 0)  # Prefer vectors with more non-zero elements
