import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pygame
from types import SimpleNamespace
import matplotlib.gridspec as gridspec

# Cấu hình project path
project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm

class HeadlessBenchmark:
    def __init__(self, num_runs=5, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []

    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game()
            bot = BotManager(game).create_bot(algorithm)

            while not game.game_over:
                state = game.get_state()
                if getattr(bot, "is_heuristic", False):
                    bullets = getattr(state, 'bullets', []) if not isinstance(state, dict) else state.get('bullets', [])
                    processed = [pygame.Vector2(float(b.x), float(b.y)) if hasattr(b, 'x') else pygame.Vector2(*map(float, b)) for b in bullets]
                    action = bot.get_action(processed)
                else:
                    action = bot.get_action(state)
                game.update(action)

            return {"algorithm": name, "run": run_idx + 1, "score": game.score}
        except:
            return None

    def run(self, algorithms):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._run_single_test, name, algo, i)
                       for name, algo in algorithms.items() for i in range(self.num_runs)]
            self.results = [f.result() for f in futures if f.result()]
        return pd.DataFrame(self.results)

def setup_environment():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except: pass
    os.environ["SDL_VIDEODRIVER"] = os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.init()
    try: pygame.display.set_mode((1, 1))
    except: pass

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    os.makedirs(base_path, exist_ok=True)
    if df.empty: return None, None

    df.to_csv(f"{base_path}/benchmark_results.csv", index=False)
    summary = df.groupby(["algorithm", "run"])["score"].mean().reset_index()

    fig = plt.figure(figsize=(14, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[5, 1])
    ax_main = fig.add_subplot(spec[0])
    ax_legend = fig.add_subplot(spec[1])
    ax_legend.axis("off")

    algorithms = summary["algorithm"].unique()
    lines, labels = [], []
    for algo in algorithms:
        data = summary[summary["algorithm"] == algo]
        line, = ax_main.plot(data["run"], data["score"], marker="o", label=algo)
        ax_main.scatter(data[data["run"].isin([10])]["run"], data[data["run"].isin([10])]["score"],
                        s=100, edgecolors='black', zorder=5)
        lines.append(line)
        labels.append(algo)

    ax_main.set_title("Algorithm Comparison", fontsize=16)
    ax_main.set_xlabel("Number of Runs", fontsize=14)
    ax_main.set_ylabel("Score", fontsize=14)
    ax_main.grid(True)
    ax_legend.legend(lines, labels, title="Thuật toán", fontsize=12, loc='center left')

    plot_path = f"{base_path}/benchmark_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

if __name__ == "__main__":
    setup_environment()
    algorithms = {
        "Furthest Safe": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        "Least Danger": DodgeAlgorithm.LEAST_DANGER_PATH,
        "Least Danger Advanced": DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        "Opposite Threat Direction": DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        "Random Safe Zone": DodgeAlgorithm.RANDOM_SAFE_ZONE,
        "DL Numpy": DodgeAlgorithm.DL_PARAM_INPUT_NUMPY,
        "DL Param Torch": DodgeAlgorithm.DL_PARAM_INPUT_TORCH,
    }

    benchmark = HeadlessBenchmark(num_runs=10, num_threads=4)
    results_df = benchmark.run(algorithms)
    plot_file = save_results(results_df)
    pygame.quit()
