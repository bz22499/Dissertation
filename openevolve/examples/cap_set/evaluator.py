# evaluator.py
import itertools
import numpy as np
import sys # Needed for debug prints
import os
import argparse # Import argparse

# Add the directory of the program to the Python path
sys.path.append(os.path.dirname(__file__))

# It is important to import the function from the generated program
# Ensure initial_program.py is in the same directory or Python path
try:
    from initial_program import solve
except ImportError:
    print("ERROR (evaluator): Could not import 'solve' from initial_program.py. Check file existence and path.")
    sys.exit(1) # Exit if import fails

def is_cap_set(vectors: np.ndarray) -> bool:
    """
    Returns whether `vectors` form a valid cap set.
    """
    # ... (Keep the rest of is_cap_set function as it was, including debug prints) ...
    print(f"DEBUG (is_cap_set): Checking validity. Input type={type(vectors)}, shape={getattr(vectors, 'shape', 'N/A')}")
    sys.stdout.flush()

    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        print(f"DEBUG (is_cap_set): Invalid type or dimensions.")
        sys.stdout.flush()
        return False

    # Handle empty cap set - it's valid
    if vectors.shape[0] == 0:
        print(f"DEBUG (is_cap_set): Empty set is valid.")
        sys.stdout.flush()
        return True

    num_vectors, n_dim = vectors.shape # Use n_dim here to avoid clash with function arg n
    print(f"DEBUG (is_cap_set): Set size={num_vectors}, dimensions={n_dim}")
    sys.stdout.flush()


    # Convert vectors to a set of tuples for efficient lookup
    vector_set = {tuple(v) for v in vectors}

    # A cap set cannot have more than 3^n_dim elements (basic check)
    if num_vectors > 3**n_dim:
        print(f"DEBUG (is_cap_set): Set too large ({num_vectors} > {3**n_dim}).")
        sys.stdout.flush()
        return False

    # Check for the cap set property
    line_found = False
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors): # Check distinct pairs i, j
            v_i = vectors[i]
            v_j = vectors[j]
            v_i_tuple = tuple(v_i)
            v_j_tuple = tuple(v_j)

            # Calculate the third point that would form a line
            v_k_tuple = tuple((-v_i - v_j) % 3)

            # Check if this third point k exists in the set
            if v_k_tuple in vector_set:
                # To be a line, all three points must be distinct.
                # Since i < j, v_i_tuple != v_j_tuple. Check k vs i and k vs j.
                if v_k_tuple != v_i_tuple and v_k_tuple != v_j_tuple:
                    print(f"DEBUG (is_cap_set): Line found! v_i={v_i_tuple}, v_j={v_j_tuple}, v_k={v_k_tuple}")
                    sys.stdout.flush()
                    line_found = True
                    break # Break inner loop
        if line_found:
            break # Break outer loop

    if line_found:
        print(f"DEBUG (is_cap_set): Result=False (line found).")
        sys.stdout.flush()
        return False
    else:
        print(f"DEBUG (is_cap_set): Result=True (no line found).")
        sys.stdout.flush()
        return True

# ### THIS IS THE MAIN FIX ###
# Change the signature to accept *args, **kwargs to catch any unexpected arguments
# But explicitly set n = 4 inside the function
def evaluate(*args, **kwargs) -> dict[str, float]:
    """
    Returns a dictionary of metrics for the `n`-dimensional cap set.
    Ignores any arguments passed and always uses n=4.
    """
    n = 4 # Explicitly set n here to override incorrect arguments
    print(f"\nDEBUG (evaluator): Starting evaluation. Forcing n={n}") # Updated debug print
    sys.stdout.flush()
    score = 0.0 # Default score
    capset = None # Initialize capset variable

    try:
        capset = solve(n) # Pass the correct integer n
        print(f"DEBUG (evaluator): Received capset from solve(): type={type(capset)}, shape={getattr(capset, 'shape', 'N/A')}")
        sys.stdout.flush()

        # Additional check: ensure return type is numpy array
        if not isinstance(capset, np.ndarray):
             print(f"DEBUG (evaluator): Error - solve() did not return a numpy array.")
             sys.stdout.flush()
             score = 0.0 # Penalize incorrect return type
        elif is_cap_set(capset):
            score = float(capset.shape[0]) # Use shape[0] for size
            print(f"DEBUG (evaluator): Capset is valid. Size={score}")
            sys.stdout.flush()
        else:
            score = 0.0
            print(f"DEBUG (evaluator): Capset is INVALID.")
            sys.stdout.flush()

    except Exception as e:
        score = 0.0
        print(f"DEBUG (evaluator): Exception during solve() or validation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for errors
        sys.stdout.flush()

    # Return a dictionary with 'combined_score'
    metrics = {
        'combined_score': score,
        'capset_size': score
    }
    print(f"DEBUG (evaluator): Returning metrics: {metrics}")
    sys.stdout.flush()
    return metrics

if __name__ == "__main__":
    # Allows running evaluator directly for testing
    # Argument parsing here is mainly for direct testing, evaluate() now ignores args
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="Dimension for cap set (only used for direct run)")
    args = parser.parse_args()

    print("--- Running evaluator directly ---")
    # Call evaluate without arguments, as it sets n internally
    metrics = evaluate()
    print(f"--- Direct run finished --- \nFinal Metrics: {metrics}")