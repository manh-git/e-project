import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from multiprocessing import Pool

from game.game_core import Game
from bot.bot_manager import BotManager

def run_single_bot(name, algorithm_enum, run_counts):
    from game.game_core import Game
    from bot.bot_manager import BotManager
    import numpy as np
    import traceback

    results = {}
    all_data = []

    for run_count in run_counts:
        scores = []
        for i in range(run_count):
            game = Game()
            bot_manager = BotManager(game)
            bot = bot_manager.create_bot(algorithm_enum)

            try:
                if getattr(bot, "is_heuristic", False) or not hasattr(bot, "train"):
                    game.run(bot, render=False)
                else:
                    game.run(bot, mode="eval", render=False)

                score = game.score
            except Exception as e:
                print(f"[{name}] lỗi ở lượt {i+1}: {e}")
                traceback.print_exc()
                score = 0

            print(f"[{name}] Game {i+1}/{run_count}: Score = {score}")
            scores.append(score)

        avg_score = np.mean(scores)
        results[run_count] = avg_score
        all_data.append({
            "algorithm": name,
            "run_count": run_count,
            "avg_score": avg_score
        })

    return name, results, all_data

class BenchmarkRunner:
    def __init__(self, run_counts=[10, 50, 100]):
        self.run_counts = run_counts
        self.results = {}

    def run(self, dodge_methods, save_csv=True, csv_filename="benchmark_result.csv",
            save_plot=True, save_path="/content/drive/MyDrive/benchmark_score_plot.png"):

        args = [(name, algo_enum, self.run_counts) for name, algo_enum in dodge_methods.items()]

        with Pool(processes=min(len(dodge_methods), os.cpu_count())) as pool:
            results_list = pool.starmap(run_single_bot, args)

        all_data = []
        for name, result, data in results_list:
            self.results[name] = result
            all_data.extend(data)

        if save_csv:
            with open(csv_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["algorithm", "run_count", "avg_score"])
                writer.writeheader()
                writer.writerows(all_data)

        if save_plot:
            df = pd.DataFrame(all_data)

            plt.figure(figsize=(12, 6))
            for algo in df["algorithm"].unique():
                subset = df[df["algorithm"] == algo]
                plt.plot(subset["run_count"], subset["avg_score"], marker='o', label=algo)

            plt.xlabel("Số lượt chơi (games)")
            plt.ylabel("Điểm trung bình")
            plt.title("So sánh hiệu năng các thuật toán tránh vật thể")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.show()


