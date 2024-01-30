from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
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


def plot_merged_graph(x1, y1, y1_smooth, label_1, x2, y2, y2_smooth, label_2, title, y_label, ylim, legend_loc,
                      x_label="Timesteps"):
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.plot(x1, y1_smooth, color='dodgerblue', label="Soft")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.1, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Aggressive")
    plt.plot(x2, y2, color='orangered', alpha=0.1, linewidth=1)
    if legend_loc:
        plt.legend(loc=legend_loc)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.grid(True, linewidth=0.6)
    return plt


if __name__ == "__main__":
    # action_spaces_rewards = (
    #     plot_multiple_graph(title="Learning curve of action spaces with all observations",
    #                         window=50, poly=1,
    #                         monitor_type="action_spaces",
    #                         x_label="timesteps",
    #                         y_label="Average reward",
    #                         leg_loc="",
    #                         x_lim=[],
    #                         y_lim=[],
    #                         variable_type="r",
    #                         variables=["continuous", "discrete_2"]))
    # action_spaces_rewards.show()
    # plt.savefig("action_spaces_rewards", dpi=300)

    # action_spaces_score = (
    #     plot_multiple_graph(title="Average episode score of action spaces with all observations",
    #                         window=50, poly=1,
    #                         monitor_type="action_spaces",
    #                         x_label="timesteps",
    #                         y_label="Average score",
    #                         leg_loc="upper left",
    #                         x_lim=[],
    #                         y_lim=[-20, 580],
    #                         variable_type="score",
    #                         variables=["continuous", "discrete_2"]))
    # plt.savefig("action_spaces_score", dpi=300)

    # Distance-based reward function
    x1, y1 = merge_runs_to_xy("monitors/base/soft", 'r')
    y1_smooth = smooth_curve(y1, 50)
    x2, y2 = merge_runs_to_xy("monitors/base/aggressive", "r")
    y2_smooth = smooth_curve(y2, 50)
    # Learning curve reward
    # distance_based_reward = plot_merged_graph(
    #     x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
    #     x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
    #     title="Learning curve using distance-based reward function",
    #     y_label="Rewards",
    #     ylim=[-2000, 4000],
    #     legend_loc="lower right",
    # )
    # Score
    # distance_based_score = plot_merged_graph(
    #     x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
    #     x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
    #     title="Episode score using distance-based reward function",
    #     y_label="Score",
    #     ylim=[-50, 1200],
    #     legend_loc="upper left",
    # )