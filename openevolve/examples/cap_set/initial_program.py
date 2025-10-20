# initial_program.py
import itertools
import numpy as np
import sys # Needed for debug prints

# --- NON-EVOLVABLE REGION ---
# The imports and function signature must remain unchanged
# so that the evaluator.py can call the 'solve' function.

def solve(n: int) -> np.ndarray:
    """
    Returns a large cap set in `n` dimensions using a structure
    similar to the FunSearch baseline greedy solver.
    This ENTIRE function body is the solver to be evolved.
    """

    ### START EVOLVABLE REGION ###

    all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

    # Powers in decreasing order for compatibility with `itertools.product`.
    powers = 3 ** np.arange(n - 1, -1, -1)

    # --- Baseline Priority Calculation ---
    # Simple zero priority for all vectors initially.
    # The LLM can replace this part entirely.
    priorities = np.zeros(len(all_vectors), dtype=float)
    # --- End Baseline Priority Calculation ---

    # --- Greedy Cap Set Construction ---
    # Build `capset` greedily, using priorities for prioritization.
    # The LLM can replace this loop structure.
    capset = np.empty(shape=(0, n), dtype=np.int32)

    # Keep track of available vector indices using a boolean mask
    # We will set priorities to -inf for unavailable vectors
    available_mask = np.ones(len(all_vectors), dtype=bool)

    iteration_count = 0 # Debug counter
    while True:
        # Find the index of the available vector with the highest priority
        current_priorities = np.where(available_mask, priorities, -np.inf)

        if np.all(current_priorities == -np.inf):
            # print(f"DEBUG (initial_program): No available vectors left after {iteration_count} iterations.") # Optional debug
            # sys.stdout.flush()
            break # No available vectors left

        max_index = np.argmax(current_priorities)
        if not available_mask[max_index]:
             # This should ideally not happen if logic is correct, but as a safeguard:
             print(f"DEBUG (initial_program): Error - max priority index {max_index} is not available!")
             sys.stdout.flush()
             break

        # Add the selected vector to the capset
        vector = all_vectors[None, max_index] # Shape (1, n)
        capset = np.concatenate([capset, vector], axis=0)

        # Mark this vector as unavailable by setting its priority to -inf
        priorities[max_index] = -np.inf
        available_mask[max_index] = False # Also update mask for clarity/safety

        # Block vectors that would form a line with the new vector
        # Calculate indices j = (-i - k) mod 3 for all i in capset (including k itself)
        # This uses the efficient FunSearch blocking logic
        if capset.shape[0] >= 1: # If capset is not empty
             # Calculate the indices of vectors that, combined with the new vector `vector`
             # and any existing vector `capset[c]`, would form a line.
             # blocking[c] = index of vector j such that capset[c] + vector + all_vectors[j] = 0 mod 3
             blocking_indices = np.einsum('cn,n->c', (-capset - vector) % 3, powers)

             # Ensure indices are valid before using them
             valid_indices = blocking_indices[blocking_indices < len(all_vectors)]

             # Mark these blocked vectors as unavailable
             priorities[valid_indices] = -np.inf
             available_mask[valid_indices] = False
        iteration_count += 1 # Debug counter

    # --- End Greedy Cap Set Construction ---

    # Final Debug Print before returning
    print(f"DEBUG (initial_program): Generated capset size: {capset.shape[0]}")
    sys.stdout.flush()

    # Ensure correct return type (already a numpy array)
    # Handle potentially empty capset (shape will be (0, n))
    return capset

    ### END EVOLVABLE REGION ###