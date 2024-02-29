from quality_metrics.common import (
    QualityMetric,
    Problem,
)
from tqdm import tqdm
from abc import abstractmethod
from typing import Tuple 



class Rank_puzzle(QualityMetric):
    def __init__(self,puzzle_dict,mode_rank="pairwise",prompt_instruction=None, n_generation=4):
        """ 
        Args:
        - puzzle_dict: a dictionary of puzzles to rank {puzzl_id: puzzle_text, ...}
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"
        - prompt_instruction: the prompt to use for the ranking
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle
        """
        
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
    
    def pairwise_ranking(self,puzzle1: str,puzzle2: str) -> int:
        """
        return the winner:
        - 0 if puzzle1 wins
        - 1 if puzzle2 wins
        - 2 if draw
        """
        raise NotImplementedError
    
    def absolute_grade(self,puzzle):
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
        keys = list(self.puzzle_dict.keys())
        grades = {key: [] for key in self.puzzle_dict}
        with tqdm(total=int(len(keys)*self.n_generation)) as pbar:
            for i in range(len(keys)):
                for _ in range(self.n_generation):
                    key = keys[i]
                    puzzle = self.puzzle_dict[key]
                    grades[key].append(self.absolute_grade(puzzle))

                    pbar.update(1)
                    if self.speed_inference!=None:
                        pbar.set_description(f"Speed Inference: {self.speed_inference} tok/s")
                        pbar.refresh()
                

        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: grades[x], reverse=True)

        return ranked_keys, grades
    
    def round_robin_tournament(self,):
        # Initialize win records for each puzzle key
        win_record = {key: 0 for key in self.puzzle_dict}
        
        # Get a list of keys to iterate over
        keys = list(self.puzzle_dict.keys())
        
        # Iterate over each unique pair of puzzle keys
        total_iter = int(len(keys)*(len(keys)-1)/2*self.n_generation)
        with tqdm(total=total_iter) as pbar:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    puzzle1, puzzle2 = self.puzzle_dict[key1], self.puzzle_dict[key2]
                    
                    # Determine the winner using the pairwise ranking function
                    for _ in range(self.n_generation):
                        res_pairwise = self.pairwise_ranking(puzzle1, puzzle2)
                        res_pairwise_inverse = self.pairwise_ranking(puzzle2, puzzle1)
                        

                        # Update the win record for the winner
                        if  res_pairwise == 0:
                            win_record[key1] += 1
                        elif res_pairwise == 1:
                            win_record[key2] += 1
                        elif res_pairwise == 2:
                            win_record[key1] += 0.5
                            win_record[key2] += 0.5
                        else: 
                            raise ValueError(f"Invalid result: {res_pairwise}")
                        
                        self.save_results.append((key1,key2,res_pairwise))
                        self.save_results_inverse.append((key2,key1,res_pairwise_inverse))
                        self.save_all_results.append((key1,key2,res_pairwise))
                        self.save_all_results.append((key2,key1,res_pairwise_inverse))
                        pbar.update(1)
                        if self.speed_inference!=None:
                            pbar.set_description(f"Speed Inference: {self.speed_inference} tok/s")
                            pbar.refresh()
        
        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: win_record[x], reverse=True)
        
        # Convert ranked keys back to their corresponding puzzle names or descriptions
        ranked_puzzles = [(key, self.puzzle_dict[key]) for key in ranked_keys]
        return ranked_puzzles, win_record
    
    def computing_ranking(self) -> Tuple[list,dict]:
        if self.mode_rank == "pairwise":
            return self.round_robin_tournament()
        elif self.mode_rank == "absolute":
            return self.absolute_ranking()
        else:
            raise ValueError(f"Invalid ranking mode: {self.mode_rank}")


