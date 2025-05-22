import multiprocessing
import os
import sys

project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bot.evaluation.mark_Runner import BenchmarkRunner
from bot.heuristic_dodge import HeuristicDodgeBot
from bot.deep_learning.param_input.numpy_agent import ParamNumpyAgent
from bot.deep_learning.param_input.pytorch_agent import ParamTorchAgent
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm 

def main():
    game = Game()

    dodge_methods = {
        "Furthest Safe Direction": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        
    }

     runner = BenchmarkRunner(run_counts=[3])

    save_path = "/content/drive/MyDrive/game_ai/benchmark_score_plot.png"
    csv_path = "/content/drive/MyDrive/game_ai/benchmark_results.csv"
    runner.run(dodge_methods=dodge_methods, save_csv=True, csv_filename=csv_path, save_plot=True, save_path=save_path)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  
    main()
