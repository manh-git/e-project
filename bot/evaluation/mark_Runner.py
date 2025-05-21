import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
from game.game_core import Game
from bot.heuristic_dodge import HeuristicDodgeBot
from bot.deep_learning.param_input.numpy_agent import ParamNumpyAgent
from bot.deep_learning.param_input.pytorch_agent import ParamTorchAgent
from configs.bot_config import DodgeAlgorithm


class BenchmarkRunner:
    def __init__(self, num_games=10):
        self.num_games = num_games
        self.results = []

    def run_single_game(self, args):
        bot_name, bot_factory, game_id = args
        game = Game()
        bot = bot_factory(game)  
        score = game.run(bot)
        return {"bot": bot_name, "game_id": game_id, "score": score}

    def run(self, bot_factories, save_csv=True, plot=True, 
            csv_filename="benchmark_results.csv", plot_filename="benchmark_plot.png"):
        tasks = [
            (bot_name, bot_factory, game_id)
            for bot_name, bot_factory in bot_factories.items()
            for game_id in range(self.num_games)
        ]

        self.results = [self.run_single_game(task) for task in tasks]


        if save_csv:
            self._save_results(csv_filename)

        if plot:
            self._plot_results(output_path=plot_filename)

    def _save_results(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bot", "game_id", "score"])
            writer.writeheader()
            writer.writerows(self.results)

    def _plot_results(self, output_path=None):
        df = pd.DataFrame(self.results)
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df, x="game_id", y="score", hue="bot", marker="o")
        plt.title("Bot Score over Multiple Games")
        plt.xlabel("Game Number")
        plt.ylabel("Score")
        plt.legend(title="Bot/Algorithm")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"✅ Đã lưu biểu đồ vào: {output_path}")
        else:
            plt.show()
