import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import pygame
import traceback
import numpy as np
from types import SimpleNamespace

# Cấu hình project path
project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import game components
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm

class HeadlessBenchmark:
    def __init__(self, num_runs=5, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []
        
   def _run_single_test(self, name, algorithm, run_idx):
    """Chạy một game không đồ họa đến khi game over"""
    try:
        game = Game()
        bot_manager = BotManager(game)

        bot_creators = {
            DodgeAlgorithm.FURTHEST_SAFE_DIRECTION: lambda: bot_manager.create_bot(DodgeAlgorithm.FURTHEST_SAFE_DIRECTION),
            DodgeAlgorithm.LEAST_DANGER_PATH: lambda: bot_manager.create_bot(DodgeAlgorithm.LEAST_DANGER_PATH),
            DodgeAlgorithm.DL_PARAM_INPUT_NUMPY: lambda: bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_NUMPY)
        }

        bot = bot_creators.get(algorithm, lambda: None)()
        if not bot:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Chạy game
        start_time = time.time()
        while True:
            state = game.get_state()

            # Chuyển player về dạng có .x và .y
            if 'player' in state:
                p = state['player']
                if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2:
                    state['player'] = SimpleNamespace(x=p[0], y=p[1])
                elif not hasattr(p, 'x') or not hasattr(p, 'y'):
                    print(f"[WARN] Player không rõ định dạng: {p}")
                    continue  # Bỏ lượt nếu không thể dùng

            # Chuyển bullets về dạng có .x và .y
            if 'bullets' in state:
                new_bullets = []
                for bullet in state['bullets']:
                    if isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) == 2:
                        new_bullets.append(SimpleNamespace(x=bullet[0], y=bullet[1]))
                    elif hasattr(bullet, 'x') and hasattr(bullet, 'y'):
                        new_bullets.append(bullet)
                    else:
                        print(f"[WARN] Bullet không rõ định dạng: {bullet}")
                state['bullets'] = new_bullets

            action = bot.get_action(state)
            game.update(action)
            if game.game_over:
                break

        return {
            "algorithm": name,
            "run": run_idx + 1,
            "score": game.score,
            "duration": time.time() - start_time
        }

    except Exception as e:
        print(f"[ERROR] {name} run {run_idx}: {str(e)}")
        traceback.print_exc()
        return None
    def run(self, algorithms):
        """Chạy benchmark cho tất cả thuật toán"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for name, algo in algorithms.items():
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_single_test, name, algo, i
                    ))

            for future in futures:
                if (result := future.result()):
                    self.results.append(result)

        return pd.DataFrame(self.results)

def setup_environment():
    print("[DEBUG] Bắt đầu setup môi trường")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[DEBUG] Mount drive thành công")
    except Exception as e:
        print(f"[WARN] Lỗi mount drive: {e}")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    pygame.init()
    try:
        pygame.display.set_mode((1, 1))
    except Exception as e:
        print(f"[WARN] Không thể tạo cửa sổ pygame: {e}")
    print("[DEBUG] pygame đã khởi tạo")

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    """Lưu kết quả và vẽ biểu đồ đường theo số lượt chơi"""
    os.makedirs(base_path, exist_ok=True)

    if df.empty:
        print("[WARN] DataFrame kết quả rỗng, không lưu được file!")
        return None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Tính điểm trung bình theo mỗi lượt chơi và thuật toán
    summary = df.groupby(["algorithm", "run"])["score"].mean().reset_index()

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))
    algorithms = summary["algorithm"].unique()

    for algo in algorithms:
        algo_df = summary[summary["algorithm"] == algo]
        plt.plot(algo_df["run"], algo_df["score"], marker="o", label=algo)

        # Tô chấm tròn lớn tại các mốc đặc biệt
        highlight_runs = [10, 20, 50, 100]
        highlight_points = algo_df[algo_df["run"].isin(highlight_runs)]
        plt.scatter(highlight_points["run"], highlight_points["score"], 
                    s=100, edgecolors='black', zorder=5)

    plt.title("Điểm trung bình theo số lượt chơi", fontsize=16)
    plt.xlabel("Số lượt chơi (Run)", fontsize=14)
    plt.ylabel("Điểm trung bình", fontsize=14)
    plt.legend(title="Thuật toán", fontsize=12)
    plt.grid(True)

    plot_path = f"{base_path}/benchmark_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return csv_path, plot_path


if __name__ == "__main__":
    print("[DEBUG] Bắt đầu chương trình chính")
    setup_environment()

    algorithms = {
        "Furthest Safe": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        "Least Danger": DodgeAlgorithm.LEAST_DANGER_PATH,
        "DL Numpy": DodgeAlgorithm.DL_PARAM_INPUT_NUMPY
    }

    benchmark = HeadlessBenchmark(num_runs=10, num_threads=4)
    results_df = benchmark.run(algorithms)

    csv_file, plot_file = save_results(results_df)

    if csv_file and plot_file:
        print("\nBenchmark hoàn tất!")
        print(f"→ Kết quả CSV: {csv_file}")
        print(f"→ Biểu đồ: {plot_file}")
        print("\nThống kê điểm số:")
        print(results_df.groupby('algorithm')['score'].describe())
    else:
        print("\nBenchmark không có kết quả để lưu!")

    pygame.quit()

