"""Finds large salem-spencer progressions, i.e. sequences of integers where no three elements form an arithmetic progression.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import numpy as np

import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of a salem-spencer set for n variables"""
  ss_set = solve(n)
  return len(ss_set)


def solve(n: int) -> np.ndarray:
  """Returns a salem-spencer set for n variables"""
  all_integers = np.arange(n)

  # Precompute all priorities.
  priorities = np.array([priority(int, n) for int in all_integers])

  # Build `set` greedily, using priorities for prioritization.
  ss_set = np.empty(shape=(0,), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a integer with maximum priority to `set`, and set priorities of
    # invalidated integers to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    pick_new_int = all_integers[None, max_index]  # [1, n]

    #identify the elements which would form part of an arithmetic progression
    blocking = 2*pick_new_int - ss_set
    blocking2= 2*ss_set - pick_new_int
    blocking3 = (pick_new_int + ss_set) / 2
    blocking3 = np.round(blocking3[np.isclose(blocking3, np.round(blocking3))]).astype(int)

    #remove those elements from the priority list
    blocking_all = np.concatenate([blocking, blocking2,blocking3], axis=0)
    blocking_all = blocking_all[(blocking_all < n) & (blocking_all >= 0)]
    priorities[blocking_all] = -np.inf
    priorities[max_index] = -np.inf
    ss_set = np.concatenate([ss_set, pick_new_int], axis=0)

  return ss_set


@funsearch.evolve
def priority(k: int, n: int) -> float:
  """Returns the priority with which we want to add `element` to the salem-spencer set.
  n is the number of possible integers, and k is the integer we want to determine priority for. 
  """
  return 0.0
