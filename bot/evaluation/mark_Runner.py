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

from game.game_core import Game
from bot.bot_manager import BotManager

def run_single_bot(name, create_bot_func, run_counts):
    results = {}
    all_data = []

    for run_count in run_counts:
        scores = []
        for i in range(run_count):
            game = Game()
            bot = create_bot_func(game)

            try:
                print(f"[{name}] Game {i+1}/{run_count}: Bắt đầu chạy game", flush=True)
                if getattr(bot, "is_heuristic", False) or not hasattr(bot, "train"):
                    game.run(bot, render=False)
                else:
                    game.run(bot, mode="eval", render=False)

                score = game.score
            except Exception as e:
                print(f"[{name}] lỗi ở lượt {i+1}: {e}", flush=True)
                traceback.print_exc()
                score = 0

            print(f"[{name}] Game {i+1}/{run_count}: Score = {score}", flush=True)
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
    def __init__(self, run_counts=[1]):
        self.run_counts = run_counts
        self.results = {}

    def run(self, dodge_methods, save_csv=True, csv_filename="benchmark_result.csv",
            save_plot=True, save_path="/content/drive/MyDrive/benchmark_score_plot.png"):

        all_data = []
        for name, create_bot_func in dodge_methods.items():
            name, result, data = run_single_bot(name, create_bot_func, self.run_counts)
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
            for name in df["algorithm"].unique():
                df_algo = df[df["algorithm"] == name]
                plt.plot(df_algo["run_count"], df_algo["avg_score"], marker='o', label=name)

            plt.xlabel("Số lượt chơi (games)")
            plt.ylabel("Điểm trung bình")
            plt.title("So sánh hiệu năng thuật toán tránh vật thể")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)
            plt.show()
