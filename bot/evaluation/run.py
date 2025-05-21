import os
import sys
project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.game_core import Game
from bot.heuristic_dodge import HeuristicDodgeBot
from bot.deep_learning.param_input.numpy_agent import ParamNumpyAgent
from bot.deep_learning.param_input.pytorch_agent import ParamTorchAgent
from configs.bot_config import DodgeAlgorithm

def create_furthest_safe_bot():
    game = Game()
    return HeuristicDodgeBot(game, DodgeAlgorithm.FURTHEST_SAFE_DIRECTION)

def create_least_danger_bot():
    game = Game()
    return HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH)

def create_least_danger_adv_bot():
    game = Game()
    return HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED)

def create_opposite_threat_bot():
    game = Game()
    return HeuristicDodgeBot(game, DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION)

def create_random_safe_bot():
    game = Game()
    return HeuristicDodgeBot(game, DodgeAlgorithm.RANDOM_SAFE_ZONE)

def create_numpy_bot():
    game = Game()
    return ParamNumpyAgent(game, load_saved_model=True)

def create_torch_bot():
    game = Game()
    return ParamTorchAgent(game, load_saved_model=True)

from bot.evaluation.mark_Runner import BenchmarkRunner

def main():
    bots = {
        "Furthest Safe Direction": create_furthest_safe_bot,
        "Least Danger": create_least_danger_bot,
        "Least Danger Advanced": create_least_danger_adv_bot,
        "Opposite Threat Direction": create_opposite_threat_bot,
        "Random Safe Zone": create_random_safe_bot,
        "DL - Numpy": create_numpy_bot,
        "DL - Torch": create_torch_bot
    }

    runner = BenchmarkRunner(num_games=10)
    runner.run(bots, save_csv=True, plot=True, csv_filename="benchmark_results.csv")

if __name__ == "__main__":
    main()

