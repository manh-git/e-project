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

from enum import Enum

class DodgeAlgorithm(Enum):
    FURTHEST_SAFE_DIRECTION = 1
    LEAST_DANGER_PATH = 2
    LEAST_DANGER_PATH_ADVANCED = 3
    RANDOM_SAFE_ZONE = 4
    OPPOSITE_THREAT_DIRECTION = 5
    DL_PARAM_INPUT_NUMPY = 6
    DL_PARAM_INPUT_TORCH = 7

def main():
    game = Game()

    dodge_methods = {
        "Furthest Safe Direction": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.FURTHEST_SAFE_DIRECTION),
        "Least Danger": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH),
        "Least Danger Advanced": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED),
        "Opposite Threat Direction": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION),
        "Random Safe Zone": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.RANDOM_SAFE_ZONE),
        "DL Param Numpy": lambda: ParamNumpyAgent(game, load_saved_model=True),
        "DL Param Torch": lambda: ParamTorchAgent(game, load_saved_model=True)
    }

    runner = BenchmarkRunner()
    save_path = "/content/drive/MyDrive/game_ai/benchmark_score_plot.png"
    runner.run(dodge_methods=dodge_methods, save_plot=True, save_path=save_path)

if __name__ == "__main__":
    main()


