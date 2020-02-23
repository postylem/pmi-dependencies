"""
Methods to get a tree structure from a distance matrix (torch tensor).
"""

import torch
import numpy as np

class GetTree:
  """Abstract class mapping matrix to tree structure"""

  @staticmethod
  def tree(matrix, words):
    """Maps matrix to tree.  
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class MST(GetTree):
  """Gets tree as MST from matrix of distances"""

  @staticmethod
  def tree(matrix, words, symmetrize_method='sum'):
    '''
    Gets Maximum Spanning Tree (list of edges) from a nonsymmetric (PMI) matrix,
    using the specified method.
    input:
      matrix: an array of PMIs
      words: a list of tokens
      symmetrize_method:
        'sum' (default): sums matrix with transpose of matrix;
        'triu': uses only the upper triangle of matrix;
        'tril': uses only the lower triangle of matrix;
        'none': uses the optimum weight for each unordered pair of edges.
    returns: tree (list of edges)
    '''

    if symmetrize_method == 'sum':
      matrix = matrix + np.transpose(matrix)
    elif symmetrize_method == 'triu':
      matrix = np.triu(matrix) + np.transpose(np.triu(matrix))
    elif symmetrize_method == 'tril':
      matrix = np.tril(matrix) + np.transpose(np.tril(matrix))
    elif symmetrize_method != 'none':
      raise ValueError("Unknown symmetrize_method. Use 'sum', 'triu', 'tril', or 'none'")

    edges = MST.prims(matrix, words, maximum_spanning_tree=True)
    return edges

  @staticmethod
  def prims(matrix, words, maximum_spanning_tree=True):
    '''
    Constructs a maximum spanning tree using Prim's algorithm.
      (set maximum_spanning_tree=False to get minumum spanning tree instead).
    Input: matrix (ndArray of PMIs), words (list of tokens)
    Excludes edges to/from punctuation symbols or empty strings, and sets np.NaN to -inf
    Returns: tree (list of edges).
    Based on code by John Hewitt.
    '''
    pairs_to_weights = {}
    excluded = ["", "'", "''", ",", ".", ";", "!", "?", ":", "``", "-LRB-", "-RRB-"]
    union_find = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
      for j_index, dist in enumerate(line):
        if words[i_index] in excluded: continue
        if words[j_index] in excluded: continue
        pairs_to_weights[(i_index, j_index)] = dist
    edges = []
    for (i_index, j_index), _ in sorted(pairs_to_weights.items(),
                                        key=lambda x: float('-inf') if (x[1] != x[1]) else x[1],
                                        reverse=maximum_spanning_tree):
      if union_find.find(i_index) != union_find.find(j_index):
        union_find.union(i_index, j_index)
        edges.append((i_index, j_index))
    return edges

class UnionFind:
  '''
  Naive UnionFind (for computing MST with Prim's algorithm).
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i, j):
    '''
    join i and j
    '''
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    '''
    find parent of i
    '''
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent
