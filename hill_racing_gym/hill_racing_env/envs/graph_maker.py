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
    fig = plt.figure(title)
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


if __name__ == "__main__":
    # plt = plot_single_graph(title="Learning curve of baseline method smoothed", label_name="Baseline",
    #                         df="monitors/base", variable_type='r', y_label="Rewards", window=200)

    # single_observations = plot_multiple_graph(title="Learning curve of single observations smoothed",
    #     window=200, poly=1, monitor_type="single_obs", variables=["angle", "ground", "position", "speed"])

    # single_observations = (
    #     plot_multiple_graph(title="Learning curve of single observations",
    #                         window=200, poly=1,
    #                         monitor_type="single_obs",
    #                         y_label="Average rewards",
    #                         leg_loc="lower right",
    #                         y_lim=[-1000, 2750],
    #                         variable_type="r",
    #                         variables=["angle", "ground", "position", "speed"]))

    # multiple_observations = (
    #     plot_multiple_graph(title="Learning curve of multiple observations",
    #                         window=50, poly=1,
    #                         monitor_type="multi_obs",
    #                         x_label="timesteps",
    #                         y_label="Average reward",
    #                         leg_loc="",
    #                         x_lim=[],
    #                         y_lim=[-2100, 2900],
    #                         variable_type="r",
    #                         variables=["position_angle", "position_ground", "position_speed", "position_angle_speed"]))
    # multiple_observations.show()
    # # plt.savefig("multiple_observations_rewards", dpi=300)
    #
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

    # action_spaces_best_rewards = (
    #     plot_multiple_graph(title="Learning curve of action spaces with position_angle observations",
    #                         window=50, poly=1,
    #                         monitor_type="action_spaces_best",
    #                         x_label="timesteps",
    #                         y_label="Average reward",
    #                         leg_loc="",
    #                         x_lim=[],
    #                         y_lim=[],
    #                         variable_type="r",
    #                         variables=["continuous_position_angle", "discrete_2_position_angle"]))
    # plt.savefig("action_spaces_position_angle_rewards", dpi=300)

    # action_spaces_best_score = (
    #     plot_multiple_graph(title="Average episode score of action spaces with position_angle observations",
    #                         window=50, poly=1,
    #                         monitor_type="action_spaces_best",
    #                         x_label="timesteps",
    #                         y_label="Average score",
    #                         leg_loc="upper left",
    #                         x_lim=[],
    #                         y_lim=[-20, 580],
    #                         variable_type="score",
    #                         variables=["continuous_position_angle", "discrete_2_position_angle"]))
    # plt.savefig("action_spaces_position_angle_score", dpi=300)
    df = load_results("base0")
    plt = plot_single_graph(df="base0", label_name="base", y_label="reward", title="", variable_type="r")
    plt.show()
