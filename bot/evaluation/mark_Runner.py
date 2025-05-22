import time
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

    def plot_results(self, save_plot=False, plot_path=None):
        if not self.results:
            print("Chưa có dữ liệu!")
            return
        
        df = pd.DataFrame(self.results)
        plt.figure(figsize=(15, 8))
        
        sns.scatterplot(
            x='run',
            y='score',
            hue='algorithm',
            data=df,
            s=100,
            alpha=0.7,
            palette='tab10'
        )
        
        sns.lineplot(
            x='run',
            y='score',
            hue='algorithm',
            data=df,
            palette='tab10',
            alpha=0.3,
            legend=False
        )
        
        plt.title("ĐIỂM SỐ KHI GAME OVER", fontsize=16)
        plt.xlabel("Lần chạy", fontsize=14)
        plt.ylabel("Điểm số", fontsize=14)
        plt.xticks(range(1, self.num_runs + 1))
        plt.grid(alpha=0.3)
        
        plt.tight_layout()

        if save_plot and plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Đã lưu biểu đồ tại: {plot_path}")

        plt.show()

# Ví dụ sử dụng (trong Google Colab, bạn gắn đúng đường dẫn Drive của bạn)
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

    benchmark = GameOverBenchmark(num_runs=5, num_threads=4)

    # Đường dẫn lưu trên Google Drive (bạn thay đổi theo thư mục của bạn)
    csv_file_path = "/content/drive/MyDrive/game_ai/benchmark_results.csv"
    plot_file_path = "/content/drive/MyDrive/game_ai/benchmark_score_plot.png"

    results_df = benchmark.run_benchmark(
        algorithms,
        save_csv=True,
        csv_path=csv_file_path
    )
    print("Kết quả chi tiết:")
    print(results_df)

    print("\nThống kê:")
    print(results_df.groupby('algorithm')['score'].describe())

    benchmark.plot_results(
        save_plot=True,
        plot_path=plot_file_path
    )
