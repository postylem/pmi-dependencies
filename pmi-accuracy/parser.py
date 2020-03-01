"""
Methods to get a tree structure from a distance matrix (torch tensor).
"""

import torch
import numpy as np

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

class DepParse:
  """Gets tree as MST from matrix of distances"""

  def __init__(self, parsetype, matrix, words):
    self.parsetype = parsetype
    self.matrix = matrix
    self.words = words

  def tree(self, symmetrize_method='sum'):
    '''
    Gets a Spanning Tree (list of edges) from a nonsymmetric (PMI) matrix,
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
    sym_matrix = self.matrix
    if symmetrize_method == 'sum':
      sym_matrix = sym_matrix + np.transpose(sym_matrix)
    elif symmetrize_method == 'triu':
      sym_matrix = torch.tensor(np.triu(sym_matrix) + np.transpose(np.triu(sym_matrix)))
    elif symmetrize_method == 'tril':
      sym_matrix = torch.tensor(np.tril(sym_matrix) + np.transpose(np.tril(sym_matrix)))
    elif symmetrize_method != 'none':
      raise ValueError("Unknown symmetrize_method. Use 'sum', 'triu', 'tril', or 'none'")

    if self.parsetype == "mst":
      edges = self.prims(sym_matrix, self.words)
    elif self.parsetype == "projective":
      edges = self.eisners(sym_matrix, self.words)
    else: 
      raise ValueError("Unknown parsetype.  Choose 'mst' or 'projective'")
    return edges

  @staticmethod
  def prims(matrix, words, maximum_spanning_tree=True):
    '''
    Constructs a maximum spanning tree using Prim's algorithm.
      (set maximum_spanning_tree=False to get minumum spanning tree instead).
    Input: matrix (torch tensor of PMIs), words (list of tokens)
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

  def eisners(self, matrix, words):
    """
    Parse using Eisner's algorithm.
    entry matrix[head][dep] of matrix is the score for the arc from head to dep
    based on DependencyDecoder class from lxmls-toolkit
    https://github.com/LxMLS/lxmls-toolkit/blob/master/lxmls/parsing/dependency_decoder.py
    """

    # with np.printoptions(precision=2, suppress=True):
    #   print(f"raw input matrix for eisners\n{matrix.numpy()}")

    excluded = ["", "'", "''", ",", ".", ";", "!", "?", ":", "``", "-LRB-", "-RRB-"]
    ignore_cond = [word not in excluded for word in words]
    wordnum_to_index = {}
    counter = 0
    for index, boolean in enumerate(ignore_cond):
      if boolean:
        wordnum_to_index[counter] = index
        counter += 1
    # print(f"wordnum_to_index: {wordnum_to_index}")

    # print(f"ignore_cond: {ignore_cond}")
    matrix = matrix[ignore_cond, :][:, ignore_cond]
    # with np.printoptions(precision=2, suppress=True):
    #   print(f"input just words matrix for eisners\n{np.array(matrix.numpy())}")

    # add a column and a row of zeros at index 0, for the root of the tree.
    # Note: 0-index is reserved for the root
    # in the algorithm, values in the first column and the main diagonal will be ignored
    # (nothing points to the root and nothing points to itself)
    # I'll fill the first row with a large negative value, to prevent more than one arc from root
    matrix = matrix.float()
    col_zeros = torch.zeros(matrix.shape[0]).reshape(-1, 1)
    matrix_paddedcol = torch.cat([col_zeros, matrix], dim=1)
    row_zeros = torch.zeros(matrix_paddedcol.shape[1]).reshape(1, -1)
    row_zeros = row_zeros.fill_(-50)
    matrix_padded = torch.cat([row_zeros, matrix_paddedcol], dim=0)

    scores = matrix_padded.numpy()
    # with np.printoptions(precision=2, suppress=True):
    #   print(f"input 'scores' for eisners\n{scores}")

    # ---- begin algorithm ------

    nrows, ncols = np.shape(scores)
    if nrows != ncols:
      raise ValueError("scores must be a nparray with nw+1 rows")

    N = nrows - 1  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1, N+1):
      for s in range(N-k+1):
        t = s + k

        # First, create incomplete items.
        # left tree
        incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
        incomplete[s, t, 0] = np.max(incomplete_vals0)
        incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
        # right tree
        incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
        incomplete[s, t, 1] = np.max(incomplete_vals1)
        incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

        # Second, create complete items.
        # left tree
        complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
        complete[s, t, 0] = np.max(complete_vals0)
        complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
        # right tree
        complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
        complete[s, t, 1] = np.max(complete_vals1)
        complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    # value = complete[0][N][1]
    heads = -np.ones(N + 1, dtype=int)
    self.eisners_backtrack(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    # ---- end algorithm -----------

    # value_proj = 0.0
    # for m in range(1, N+1):
    #   h = heads[m]
    #   value_proj += scores[h, m]

    edgelist = [enumerate(heads)]
    # Eisner edges, sorted, removing the root node (taking indices [2:] and shifting all values -1)
    sortededges_noroot = sorted({tuple(sorted(tuple([i-1 for i in edge]))) for edge in edgelist})[2:]
    # Now with indices translated to give word-to-word edges (simply skipping puncuation indices)
    edges = [tuple(wordnum_to_index[w] for w in pair) for pair in sortededges_noroot]
    return edges

  def eisners_backtrack(self, incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    """
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    """
    if s == t:
      return
    if complete:
      r = complete_backtrack[s][t][direction]
      if direction == 0:
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
        return
      else:
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
        return
    else:
      r = incomplete_backtrack[s][t][direction]
      if direction == 0:
        heads[s] = t
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
        return
      else:
        heads[t] = s
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
        return
