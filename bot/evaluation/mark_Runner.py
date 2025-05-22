import time
import sys

project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import pygame
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm
import os


class GameOverBenchmark:
    def __init__(self, num_runs=5, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []

    def _run_until_game_over(self, name, algorithm, run_idx):
        try:
            pygame.init()
            screen = pygame.display.set_mode((1280, 720))
            game = Game()
            bot_manager = BotManager(game)
            
            if isinstance(algorithm, DodgeAlgorithm):
                bot = bot_manager.create_bot(algorithm)
            elif algorithm == "DL_NUMPY":
                bot = bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_NUMPY)
            elif algorithm == "DL_TORCH":
                bot = bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_TORCH)

            start_time = time.time()
            clock = pygame.time.Clock()
            
            while True:
                clock.tick(60)
                state = game.get_state()
                action = bot.get_action(state)
                game.update(action)
                if game.game_over:
                    break
                game.draw()
                pygame.display.flip()
            
            result = {
                "algorithm": name,
                "run": run_idx + 1,
                "score": game.score,
                "duration": time.time() - start_time
            }
            pygame.quit()
            return result
        except Exception as e:
            print(f"Lỗi {name} lần {run_idx}: {str(e)}")
            pygame.quit()
            return None

    def run_benchmark(self, algorithms, save_csv=False, csv_path=None):
        self.results = []  # reset mỗi lần chạy
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for name, algo in algorithms.items():
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_until_game_over, name, algo, i
                    ))
            for future in futures:
                result = future.result()
                if result:
                    self.results.append(result)
        
        df = pd.DataFrame(self.results)
        if save_csv and csv_path is not None:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Đã lưu kết quả CSV tại: {csv_path}")
        return df

    @staticmethod
    def plot_results(df, save_plot=False, plot_path=None):
        if df.empty:
            print("Chưa có dữ liệu!")
            return

        plt.figure(figsize=(15, 8))
        
        # Tính điểm trung bình theo algorithm và num_runs
        avg_scores = df.groupby(['algorithm', 'num_runs'])['score'].mean().reset_index()

        sns.lineplot(
            x='num_runs',
            y='score',
            hue='algorithm',
            data=avg_scores,
            marker='o',
            palette='tab10'
        )
        
        plt.title("Điểm trung bình theo số lượt chơi (num_runs)", fontsize=16)
        plt.xlabel("Số lượt chơi (num_runs)", fontsize=14)
        plt.ylabel("Điểm trung bình", fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()

        if save_plot and plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Đã lưu biểu đồ tại: {plot_path}")

        plt.show()


if __name__ == "__main__":
    algorithms = {
        "Furthest Safe": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        "Least Danger": DodgeAlgorithm.LEAST_DANGER_PATH,
        "Least Danger Adv": DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        "Random Safe": DodgeAlgorithm.RANDOM_SAFE_ZONE,
        "Opposite Threat": DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        "DL Numpy": "DL_NUMPY",
        "DL PyTorch": "DL_TORCH"
    }

    num_runs_list = [10, 20, 50, 100]
    all_results = []

    for runs in num_runs_list:
        print(f"Chạy benchmark với num_runs = {runs} ...")
        benchmark = GameOverBenchmark(num_runs=runs, num_threads=4)
        df = benchmark.run_benchmark(algorithms, save_csv=False)
        df['num_runs'] = runs  # thêm cột số lượt chạy
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    # Thay đường dẫn thành của bạn
    csv_file_path = "/content/drive/MyDrive/game_ai/benchmark_results.csv"
    plot_file_path = "/content/drive/MyDrive/game_ai/benchmark_score_plot.png"

    # Lưu file CSV tổng hợp
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    final_df.to_csv(csv_file_path, index=False)
    print(f"Đã lưu tổng kết quả CSV tại: {csv_file_path}")

    # Vẽ và lưu biểu đồ
    GameOverBenchmark.plot_results(final_df, save_plot=True, plot_path=plot_file_path)
