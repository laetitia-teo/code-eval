from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import math
import pandas as pd
def plot_pairwise_comparison_results(puzzle_dict,save_results,title=''):
    # Create an empty grid dataframe
    grid = pd.DataFrame(index=puzzle_dict.keys(), columns=puzzle_dict.keys())
    # Fill the grid with pairwise comparison results
    loss_value=-1
    draw_value=0
    win_value=1
    for key1, key2, res_pairwise in save_results:
        if math.isnan(grid.loc[key1, key2]):
            grid.loc[key1, key2] = 0
        if math.isnan(grid.loc[key2, key1]):
            grid.loc[key2, key1] = 0
        if res_pairwise == 0:
            grid.loc[key1, key2] += win_value#'Win'
            grid.loc[key2, key1] += loss_value#'Loss'
        elif res_pairwise == 1:
            grid.loc[key1, key2] += loss_value#'Loss'
            grid.loc[key2, key1] += win_value#'Win'
        elif res_pairwise == 2:
            grid.loc[key1, key2] += draw_value#'Draw'
            grid.loc[key2, key1] += draw_value#'Draw'

    order=grid.sum().sort_values(ascending=True).index
    grid_order=grid.loc[order, order]
    grid_order=grid_order.to_numpy()
    grid_order= np.nan_to_num(grid_order)
    grid_order = np.array(grid_order, dtype=float)
    plt.figure(dpi=140)
    plt.imshow(grid_order, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Pairwise comparison results'+title)
    plt.xlabel('Puzzle index')
    plt.ylabel('Puzzle index')
    # plot keys
    plt.xticks(np.arange(len(order)), order, rotation=60, ha='right', fontsize=5)
    plt.yticks(np.arange(len(order)), order,fontsize=5)
    # for i in range(len(order)):
    #     for j in range(len(order)):
    #         plt.annotate(str(grid_order[i, j]), xy=(j, i), ha='center', va='center', color='white')

    plt.show()

def extract_rank(save_all_results,puzzle_dict):

    keys = list(puzzle_dict.keys())
    win_record = {key: 0 for key in puzzle_dict}
    for idx_res in range(len(save_all_results)):
        key1, key2,res_pairwise = save_all_results[idx_res]
        if  res_pairwise == 0:
            win_record[key1] += 1
        elif res_pairwise == 1:
            win_record[key2] += 1
        elif res_pairwise == 2:
            win_record[key1] += 0.5
            win_record[key2] += 0.5
        else: 
            raise ValueError(f"Invalid result: {res_pairwise}")


    # Rank puzzles based on their win records, sorting by wins descending
    ranked_keys = sorted(keys, key=lambda x: win_record[x], reverse=True)

    # Convert ranked keys back to their corresponding puzzle names or descriptions
    ranked_puzzles = [(key, puzzle_dict[key]) for key in ranked_keys]
    sorted_win_record=sorted(win_record, key=win_record.get, reverse=True)
    dic_sorted_win_record={k: win_record[k] for k in sorted_win_record}

    return ranked_puzzles, dic_sorted_win_record,win_record

def get_rank(grades, tie=True):
    """
    Assign ranks to a list of grades, handling ties or not.
    """
    # Pair each grade with its index in the original list
    indexed_grades = list(enumerate(grades))
    # Sort the grades in descending order, keeping track of the original indices
    sorted_grades = sorted(indexed_grades, key=lambda x: x[1], reverse=True)
    # Initialize an empty list of the same length as grades to store ranks
    ranks = [0] * len(grades)
    
    if tie:  # Handling ties by giving the same rank
        rank = 1
        for i in range(len(sorted_grades)):
            if i > 0 and sorted_grades[i][1] == sorted_grades[i-1][1]:
                ranks[sorted_grades[i][0]] = ranks[sorted_grades[i-1][0]]
            else:
                ranks[sorted_grades[i][0]] = rank
            rank += 1
    else:  # No ties, consecutive ranks
        # This will be the actual rank assigned to each student, starting from 1
        actual_rank = 1
        # This will be incremented for each student, but not used directly if there are no ties
        for i in range(len(sorted_grades)):
            ranks[sorted_grades[i][0]] = actual_rank
            actual_rank += 1

    return ranks




class RankingSimilarity:
    """
    from https://github.com/dlukes/rbo
    This class will include some similarity measures between two different
    ranked lists.
    """
    def __init__(
        self,
        S: Union[List, np.ndarray],
        T: Union[List, np.ndarray],
        verbose: bool = False,
    ) -> None:
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]

        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose: If True, print out intermediate results. Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.

        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p

    def _bound_range(self, value: float) -> float:
        """Bounds the value to [0.0, 1.0]."""

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(
        self,
        k: Optional[float] = None,
        p: float = 1.0,
        ext: bool = False,
    ) -> float:
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p = 1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentially control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more items. See Eq. (21) for quantitative measure.

        Args:
            k: The depth of evaluation.
            p: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext: If True, we will extrapolate the rbo, as in Eq. (23).

        Returns:
            The rbo at depth k (or extrapolated beyond).
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)

        return self._bound_range(AO[-1])