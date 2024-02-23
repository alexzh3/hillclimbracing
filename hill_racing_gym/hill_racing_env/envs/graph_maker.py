from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Callable, List, Optional, Tuple
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def smooth_curve(y, window, poly=1):
    return savgol_filter(y, window, poly)


def transfer_to_xy(data_frame: pd.DataFrame, x_axis: str, y_var_values) -> Tuple[np.ndarray, np.ndarray]:
    if x_axis == "timesteps":
        x_var = np.cumsum(data_frame.l.values)
        y_var = y_var_values
    elif x_axis == "episodes":
        x_var = np.arange(len(data_frame))
        y_var = y_var_values
    elif x_axis == "walltime_hrs":
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = y_var_values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_single_graph(title, df, label_name, variable_type, y_label, window=50, poly=1, x_label_type="timesteps"):
    df = load_results(df)
    x, y = transfer_to_xy(df, x_label_type, df[variable_type].values)  # for rewards
    y = smooth_curve(y, window=window, poly=poly)
    # Truncate x
    # x = x[len(x) - len(y):]
    if len(x) != len(y):
        return f"Length x:{len(x)} and Length y: {len(y)}"

    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel(f"Number of {x_label_type}")
    plt.ylabel(y_label)
    plt.plot(x, y, label=label_name)
    plt.legend(loc='lower right')
    return plt


def plot_multiple_graph(monitor_type, variables, x_lim, y_lim, leg_loc, title, variable_type='r', x_label="timesteps",
                        y_label="Rewards",
                        window=50, poly=1):
    # Load in results in df
    df = {}
    df["baseline"] = load_results("monitors/base")
    for obs in variables:
        df[obs] = load_results(f"monitors/{monitor_type}/{obs}")

    # Transfer results to x and y
    x = {}
    y = {}
    for obs in df:
        x[obs], y[obs] = transfer_to_xy(df[obs], x_label, df[obs][variable_type].values)
        y[obs] = smooth_curve(y[obs], window, poly)

    # Plot results
    plt.title(title)
    plt.xlabel(f"Number of {x_label}")
    plt.ylabel(y_label)
    for obs in df:
        plt.plot(x[obs], y[obs], label=obs.capitalize())
    if leg_loc:
        plt.legend(loc=leg_loc, fontsize="8.3")
    else:
        plt.legend(loc='lower right', fontsize="8.3")
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    return plt


def merge_runs_to_xy(path, variable_type):
    merged_x = np.empty(0)
    merged_y = np.empty(0)
    for k in range(5):
        data = load_results(f"{path}/{k}")
        print(f'{path}/{k}')
        print(data['r'].nlargest(5))
        x, y = transfer_to_xy(data, "timesteps", data[variable_type].values)
        merged_x = np.concatenate((merged_x, x))
        merged_y = np.concatenate((merged_y, y))
    merged_xy = np.column_stack((merged_x, merged_y))  # Make (x,y) pairs in array
    sorted_indices = np.argsort(merged_xy[:, 0])  # Get array indices on timesteps from low to high
    merged_xy = merged_xy[sorted_indices]  # Sort the array on timesteps
    x, y = np.split(merged_xy, 2, axis=1)  # Split to x and y variables again
    x = x.flatten()  # Convert to 1D array
    y = y.flatten()
    return x, y


def plot_merged_graph(ax, x1, y1, y1_smooth, label_1, x2, y2, y2_smooth, label_2, title, y_label, ylim=[],
                      legend_loc=[],
                      x_label="Timesteps"):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(x1, y1_smooth, color='dodgerblue', label=label_1)
    ax.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    ax.plot(x2, y2_smooth, color='orangered', label=label_2)
    ax.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    if legend_loc:
        ax.legend(loc=legend_loc, framealpha=0.6)
    else:
        ax.legend()
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, linewidth=0.6)
    return ax


def graph_rewards_distance():
    # Distance-based reward function
    # x1, y1 = merge_runs_to_xy("monitors/reward_type/300/soft/distance", variable_type)
    # y1_smooth = smooth_curve(y1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust wspace as needed

    for i, variable_type in enumerate(['r', 'l', 'score']):
        x1, y1 = merge_runs_to_xy("monitors/cont_reward_type/300/soft/distance", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/cont_reward_type/300/aggressive/distance", variable_type)
        y2_smooth = smooth_curve(y2, 100)

        if variable_type == "r":
            plot_merged_graph(
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Learning curve using distance-based reward function",
                y_label="Rewards",
                legend_loc="upper left",
                ax=axes[i]
            )
        elif variable_type == "score":
            plot_merged_graph(
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode score using distance-based reward function",
                y_label="Score",
                legend_loc="upper left",
                ax=axes[i]
            )
        elif variable_type == "l":
            plot_merged_graph(
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode length using distance-based reward function",
                y_label="Episode length (in timesteps)",
                legend_loc="upper left",
                ax=axes[i]
            )
    fig.tight_layout(pad=2)


def graph_rewards_action():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed

    for i, variable_type in enumerate(['r', 'l', 'score']):
        x1, y1 = merge_runs_to_xy("monitors/reward_type/300/soft/action", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/reward_type/300/aggressive/action", variable_type)
        y2_smooth = smooth_curve(y2, 100)

        if variable_type == "r":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Learning curve using action-based reward function",
                y_label="Rewards",
                ylim=[-200, 2000],
                legend_loc="upper left"
            )
        elif variable_type == "score":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode score using action-based reward function",
                y_label="Score",
                legend_loc="upper left"
            )
        elif variable_type == "l":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode length using action-based reward function",
                y_label="Episode length (in timesteps)",
                ylim=[-100, 4000],
                legend_loc="upper left"
            )
    fig.tight_layout(pad=2)


def graph_rewards_wheel_speed():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed

    for i, variable_type in enumerate(['r', 'l', 'score']):
        x1, y1 = merge_runs_to_xy("monitors/cont_reward_type/300/aggressive/wheel_speed", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/cont_reward_type/300/soft/wheel_speed", variable_type)
        y2_smooth = smooth_curve(y2, 100)

        if variable_type == "r":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Aggressive",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Soft",
                title="Learning curve using wheel speed-based reward function",
                y_label="Rewards",
                ylim=[-200, 8000],
                legend_loc="upper left"
            )
        elif variable_type == "score":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Aggressive",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Soft",
                title="Episode score using wheel speed-based reward function",
                y_label="Score",
                legend_loc="upper left"
            )
        elif variable_type == "l":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Aggressive",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Soft",
                title="Episode length using wheel speed-based reward function",
                y_label="Episode length (in timesteps)",
                ylim=[-100, 5000],
                legend_loc="upper left"
            )
    fig.tight_layout(pad=2)


def shaping_comparison_reward():
    # Distance-based reward function
    x1, y1 = merge_runs_to_xy("monitors/reward_type/300/soft/distance", 'r')
    y1_smooth = smooth_curve(y1, 100)
    x2, y2 = merge_runs_to_xy("monitors/reward_type/300/soft/action", 'r')
    y2_smooth = smooth_curve(y2, 100)
    x3, y3 = merge_runs_to_xy("monitors/reward_type/300/soft/wheel_speed", 'r')
    y3_smooth = smooth_curve(y3, 100)

    plt.title("Learning curve of different reward functions")
    plt.ylabel("Reward")
    plt.xlabel("Timesteps")

    plt.plot(x1, y1_smooth, color='dodgerblue', label="Distance-based")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Action-based")
    plt.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    plt.plot(x3, y3_smooth, color='darkviolet', label="Speed-based")
    plt.plot(x3, y3, color='darkviolet', alpha=0.15, linewidth=1)
    plt.legend(framealpha=0.6, loc='upper left')
    plt.ylim(-2000, 6000)
    plt.grid(True, linewidth=0.6)
    return plt


def shaping_comparison_score():
    x1, y1 = merge_runs_to_xy("monitors/reward_type/300/soft/distance", 'score')
    y1_smooth = smooth_curve(y1, 100)
    x2, y2 = merge_runs_to_xy("monitors/reward_type/300/soft/action", 'score')
    y2_smooth = smooth_curve(y2, 100)
    x3, y3 = merge_runs_to_xy("monitors/reward_type/300/soft/wheel_speed", 'score')
    y3_smooth = smooth_curve(y3, 100)

    plt.title("Episode score of different reward functions")
    plt.ylabel("Score")
    plt.xlabel("Timesteps")

    plt.plot(x1, y1_smooth, color='dodgerblue', label="Distance-based")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Action-based")
    plt.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    plt.plot(x3, y3_smooth, color='purple', label="Speed-based")
    plt.plot(x3, y3, color='purple', alpha=0.15, linewidth=1)
    plt.legend(framealpha=0.6)
    plt.ylim(-50, 1000)
    plt.grid(True, linewidth=0.6)
    return plt


def shaping_comparison_length():
    x1, y1 = merge_runs_to_xy("monitors/reward_type/300/soft/distance", 'l')
    y1_smooth = smooth_curve(y1, 100)
    x2, y2 = merge_runs_to_xy("monitors/reward_type/300/soft/action", 'l')
    y2_smooth = smooth_curve(y2, 100)
    x3, y3 = merge_runs_to_xy("monitors/reward_type/300/soft/wheel_speed", 'l')
    y3_smooth = smooth_curve(y3, 100)

    plt.title("Episode length of different reward functions")
    plt.ylabel("Episode length (in timesteps)")
    plt.xlabel("Timesteps")

    plt.plot(x1, y1_smooth, color='dodgerblue', label="Distance-based")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Action-based")
    plt.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    plt.plot(x3, y3_smooth, color='purple', label="Speed-based")
    plt.plot(x3, y3, color='purple', alpha=0.15, linewidth=1)
    plt.legend(framealpha=0.6)
    plt.ylim(-100, 8200)
    plt.grid(True, linewidth=0.6)
    return plt


def make_boxplot_score():
    # High score boxplot
    x, y = merge_runs_to_xy("monitors/reward_type/300/soft/distance", 'score')
    x, y2 = merge_runs_to_xy("monitors/reward_type/300/aggressive/distance", 'score')
    x, y3 = merge_runs_to_xy("monitors/reward_type/300/soft/action", 'score')
    x, y4 = merge_runs_to_xy("monitors/reward_type/300/aggressive/action", 'score')
    x, y5 = merge_runs_to_xy("monitors/reward_type/300/soft/wheel_speed", 'score')
    x, y6 = merge_runs_to_xy("monitors/reward_type/300/aggressive/wheel_speed", 'score')
    reward_types = ["Distance soft", "Distance aggressive",
                    "Action soft", "Action aggressive",
                    "Speed soft", "Speed aggressive"]
    data = [y, y2, y3, y4, y5, y6]
    plt.title("Episode score of different reward functions")
    plt.xlabel("Score")
    plot = plt.boxplot(x=data, labels=reward_types, flierprops=dict(alpha=0.5), vert=0, patch_artist=True)
    # fill with colors
    colors = ['pink', 'pink', 'lightblue', 'lightblue', 'lightgreen', 'lightgreen']
    for patch, color in zip(plot['boxes'], colors):
        patch.set_facecolor(color)
    plt.xlim(-50, 1000)
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    graph_rewards_wheel_speed()
    plt.savefig("300_cont_wheel_speed_merged", dpi=300)
    plt.show()
