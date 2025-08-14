import numpy as np
import matplotlib.pyplot as plt









if __name__ == "__main__":
    exps = 24
    runs = 10
    labels = ["average_clipped_reward", "average_reset", "average_reward", "episodic_clipped_return", "episodic_return", "eval_average_clipped_reward", "eval_average_reset", "eval_average_reward"]
    algos = ["DDPG", "TD3", "SAC", "PPO"]
    for label in labels:
        for i in range(exps):
            for algo_idx, algo in enumerate(algos):
                data = np.load(f"outputs/{algo_idx}_{label}.npy")
                exp_indices = [i + j * exps for j in range(runs)]
                exp_data = data[:, exp_indices]
                mean_data = np.mean(exp_data, axis=1)
                standard_error_data = np.std(exp_data, axis=1) / np.sqrt(runs)
                plt.plot(mean_data, label=f"{algo}")
                plt.fill_between(range(len(mean_data)), mean_data - standard_error_data, mean_data + standard_error_data, alpha=0.2)
            plt.legend()
            plt.savefig(f"plots/Exp_{i}_{label}.pdf")
            plt.close()
                