from quality_metrics.common import (
    QualityMetric,
    Problem,
)
from tqdm import tqdm,trange
from abc import abstractmethod
from typing import Tuple 

# TODO: maybe add something a la Auto HF class
# e;g; AUTO_yes_model(backend = "openai|HF") to use judge_hf or judge_openai

class Rank_puzzle(QualityMetric):
    def __init__(self,puzzle_dict,mode_rank="pairwise",prompt_instruction=None, n_generation=4,bs=1):
        """ 
        Args:
        - puzzle_dict: a dictionary of puzzles to rank {puzzl_id: puzzle_text, ...}
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"
        - prompt_instruction: the prompt to use for the ranking
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle
        """
        self.bs = bs
        self.prompt_instruction = prompt_instruction
        self.mode_rank = mode_rank
        self.puzzle_dict = puzzle_dict
        self.save_results = []
        if mode_rank == "pairwise":
            self.save_results_inverse = []
        self.speed_inference = None
        self.list_speed_inference = []
        self.n_generation = n_generation
        self.save_all_results = []
        self.init_model()
        
    @abstractmethod
    def init_model(self):
        raise NotImplementedError
    
    def pairwise_ranking(self,list_puzzles: list[list[str,str]]) -> int:
        """
        return the winner:
        - 0 if puzzle1 wins
        - 1 if puzzle2 wins
        - 2 if draw
        Args
        - list_puzzles: a list of list of two puzzles to compare
        [[puz1,puz2],[pu3,pu4],...] -> [0,2,1,...] 
        """
        raise NotImplementedError
    
    def absolute_grade(self,list_puzzles):
        """return the absolute_grade int or float"""
        raise NotImplementedError
    
    def __call__(self, problem: Problem, problem2: Problem):
        if self.mode_rank == "pairwise":
            return self.pairwise_ranking(problem.get_problem(),problem2.get_problem())
        
        elif self.mode_rank == "absolute":
            return self.absolute_ranking(problem.get_problem())
        else:
            raise ValueError(f"Invalid ranking mode: {self.mode_rank}")
        
    
    def absolute_ranking(self):
        """
        return the ranking of the puzzles
        """
        # Get a list of keys to iterate over
        list_key_puzzle_to_rank = []
        list_puzzles_to_rank = []
        keys = list(self.puzzle_dict.keys())
        grades = {key: [] for key in self.puzzle_dict}

        # extract all pormpts
        
        for i in range(len(keys)):
            for _ in range(self.n_generation):
                key = keys[i]
                puzzle = self.puzzle_dict[key]
                list_puzzles_to_rank.append(puzzle)
                list_key_puzzle_to_rank.append(key)

        # grades all prompts
        for i in trange(0,len(keys),self.bs):
            list_puzzles_to_rank_bs = list_puzzles_to_rank[i:i+self.bs]
            list_key_puzzle_to_rank_bs = list_key_puzzle_to_rank[i:i+self.bs]
            out = self.absolute_grade(list_puzzles_to_rank_bs)
            for j in range(len(out)):
                grades[list_key_puzzle_to_rank_bs[j]].append(out[j])                

        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: grades[x], reverse=True)
        return ranked_keys, grades
    
    
    def round_robin_tournament(self,):
        list_key_puzzle_to_rank = []
        list_puzzles_to_rank = []
        # Initialize win records for each puzzle key
        win_record = {key: 0 for key in self.puzzle_dict}
        # Get a list of keys to iterate over
        keys = list(self.puzzle_dict.keys())
        
        # Iterate over each unique pair of puzzle keys
        # total_iter = int(len(keys)*(len(keys)-1)/2*self.n_generation)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1, key2 = keys[i], keys[j]
                puzzle1, puzzle2 = self.puzzle_dict[key1], self.puzzle_dict[key2]

                # Determine the winner using the pairwise ranking function
                for _ in range(self.n_generation):
                    # pairwise ranking is asymmetric, so we need to test it both ways
                    # (because of postion bias in the ranking)
                    list_puzzles_to_rank.append([puzzle1, puzzle2])
                    list_key_puzzle_to_rank.append([key1, key2])

                    list_puzzles_to_rank.append([puzzle2, puzzle1])
                    list_key_puzzle_to_rank.append([key2, key1])

        # grades all prompts
        for i in trange(0,len(list_key_puzzle_to_rank),self.bs):
            list_puzzles_to_rank_bs = list_puzzles_to_rank[i:i+self.bs]
            list_key_puzzle_to_rank_bs = list_key_puzzle_to_rank[i:i+self.bs]
            results_pairwise = self.pairwise_ranking(list_puzzles_to_rank_bs)
            for idx_res in range(len(results_pairwise)):
                res_pairwise = results_pairwise[idx_res]
                key1, key2 = list_key_puzzle_to_rank_bs[idx_res]
                if  res_pairwise == 0:
                    win_record[key1] += 1
                elif res_pairwise == 1:
                    win_record[key2] += 1
                elif res_pairwise == 2:
                    win_record[key1] += 0.5
                    win_record[key2] += 0.5
                else: 
                    raise ValueError(f"Invalid result: {res_pairwise}")
                if (i+idx_res)%2==0:
                    self.save_results.append((key1,key2,res_pairwise))
                else:
                    self.save_results_inverse.append((key1,key2,res_pairwise))

                self.save_all_results.append((key1,key2,res_pairwise))

        
        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: win_record[x], reverse=True)
        
        # Convert ranked keys back to their corresponding puzzle names or descriptions
        ranked_puzzles = [(key, self.puzzle_dict[key]) for key in ranked_keys]
        sorted_win_record=sorted(win_record, key=win_record.get, reverse=True)
        dic_sorted_win_record={k: win_record[k] for k in sorted_win_record}

        return ranked_puzzles, win_record
    

    def elo_ranking(self,factor_num_comparisons=5):
        """
        Compute the elo of the puzzles
        number of comparison = factor_num_comparisons * n_puzzles
        """
        import random

        num_comparisons= len(self.puzzle_dict)*factor_num_comparisons
        
        # function to update the elo rating
        def update_elo_rating(rating_a, rating_b, outcome, k=32):
            # Calculate expected outcome
            expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            # Actual outcome
            if outcome == "win":
                score_a = 1
            elif outcome == "lose":
                score_a = 0
            else:  # tie
                score_a = 0.5
            
            # Update rating
            new_rating_a = rating_a + k * (score_a - expected_a)
            return new_rating_a
        
        list_key_puzzle_to_rank = []
        list_puzzles_to_rank = []
        # Initialize win records for each puzzle key
        if hasattr(self,"elo_rating"):
            elo_rating = self.elo_rating
        else:
            elo_rating = {key: 1500 for key in self.puzzle_dict}
        # Get a list of keys to iterate over
        keys = list(self.puzzle_dict.keys())
        
        # Iterate over each unique pair of puzzle keys
        # total_iter = int(len(keys)*(len(keys)-1)/2*self.n_generation)
        for _ in range(num_comparisons):
            key1, key2 = random.sample(keys, 2)
            puzzle1, puzzle2 = self.puzzle_dict[key1], self.puzzle_dict[key2]

            # pairwise ranking is asymmetric, so we need to test it both ways (maybe later)
            # (because of postion bias in the ranking)
            list_puzzles_to_rank.append([puzzle1, puzzle2])
            list_key_puzzle_to_rank.append([key1, key2])

            # list_puzzles_to_rank.append([puzzle2, puzzle1])
            # list_key_puzzle_to_rank.append([key2, key1])

        for i in trange(0,len(list_key_puzzle_to_rank),self.bs):

                    
            # Perform the fake pairwise comparison
            list_puzzles_to_rank_bs = list_puzzles_to_rank[i:i+self.bs]
            list_key_puzzle_to_rank_bs = list_key_puzzle_to_rank[i:i+self.bs]
            outcomes = self.pairwise_ranking(list_puzzles_to_rank_bs)
            for idx_outcome in range(len(outcomes)):
                outcome = outcomes[idx_outcome]
                key_a, key_b = list_key_puzzle_to_rank_bs[idx_outcome]
                self.save_all_results.append((key_a, key_b,outcome))
            # Update ratings based on the comparison outcome
                if outcome != 2:  # In case of tie, no need to update both as no one wins
                    if outcome == 1:
                        elo_rating[key_a] = update_elo_rating(elo_rating[key_a], elo_rating[key_b], "win")
                        elo_rating[key_b] = update_elo_rating(elo_rating[key_b], elo_rating[key_a], "lose")
                    else:
                        elo_rating[key_a] = update_elo_rating(elo_rating[key_a], elo_rating[key_b], "lose")
                        elo_rating[key_b] = update_elo_rating(elo_rating[key_b], elo_rating[key_a], "win")

            #TODO: we should adjust the selection logic to focus on puzzles with similar ratings
                        
        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: elo_rating[x], reverse=True)
        ranked_puzzles = [(key, self.puzzle_dict[key]) for key in ranked_keys]
        # return puzzles
        self.elo_rating = elo_rating
        return ranked_puzzles, elo_rating
        
    def computing_ranking(self) -> Tuple[list,dict]:
        if self.mode_rank == "pairwise":
            return self.round_robin_tournament()
        elif self.mode_rank == "absolute":
            return self.absolute_ranking()
        else:
            raise ValueError(f"Invalid ranking mode: {self.mode_rank}")


