import os
import sys
project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bot.evaluation.mark_Runner import BenchmarkRunner
from bot.heuristic_dodge import HeuristicDodgeBot
from bot.deep_learning.param_input.numpy_agent import ParamNumpyAgent
from bot.deep_learning.param_input.pytorch_agent import ParamTorchAgent
from configs.bot_config import DodgeAlgorithm

def create_heuristic_bot(method):
    return lambda game: HeuristicDodgeBot(game, method)

def create_numpy_bot():
    return lambda game: ParamNumpyAgent(game, load_saved_model=True)

def create_torch_bot():
    return lambda game: ParamTorchAgent(game, load_saved_model=True)

def main():
    bots = {
        "Furthest Safe Direction": create_heuristic_bot(DodgeAlgorithm.FURTHEST_SAFE_DIRECTION),
        "Least Danger": create_heuristic_bot(DodgeAlgorithm.LEAST_DANGER_PATH),
        "Least Danger Advanced": create_heuristic_bot(DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED),
        "Opposite Threat Direction": create_heuristic_bot(DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION),
        "Random Safe Zone": create_heuristic_bot(DodgeAlgorithm.RANDOM_SAFE_ZONE),
        "DL - Numpy": create_numpy_bot(),
        "DL - Torch": create_torch_bot()
    }

    runner = BenchmarkRunner(num_games=10)
    runner.run(bots, save_csv=True, plot=True, csv_filename="benchmark_results.csv")

if __name__ == "__main__":
    main()
