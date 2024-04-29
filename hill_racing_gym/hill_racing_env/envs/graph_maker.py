import ast

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


def merge_runs_to_xy(path, variable_type):
    merged_x = np.empty(0)
    merged_y = np.empty(0)
    for k in range(5):
        data = pd.read_csv(f"{path}_{k}.monitor.csv", header=1)
        # data = load_results(f"{path}/{k}")
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


def merge_runs_dir_to_xy(path, variable_type):
    data = load_results(path)
    if variable_type == "position_list":
        data[variable_type] = data[variable_type].apply(ast.literal_eval)
        sorted_list = sorted(data[variable_type].sum(), key=lambda x: x[1])
        position, timestep = zip(*sorted_list)
        return position, timestep
    else:
        x, y = transfer_to_xy(data, "episodes", data[variable_type].values)
        merged_xy = np.column_stack((x, y))  # Make (x,y) pairs in array
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


def plot_merged_graph_4(ax, x1, y1, y1_smooth, label_1, x2, y2, y2_smooth, label_2, x3, y3, y3_smooth, label_3,
                        x4, y4, y4_smooth, label_4, title, y_label, ylim=[], legend_loc=[], x_label="Timesteps"):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(x1, y1_smooth, color='dodgerblue', label=label_1)
    ax.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    ax.plot(x2, y2_smooth, color='orangered', label=label_2)
    ax.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    ax.plot(x3, y3_smooth, color='purple', label=label_3)
    ax.plot(x3, y3, color='purple', alpha=0.15, linewidth=1)
    ax.plot(x4, y4_smooth, color='lime', label=label_4)
    ax.plot(x4, y4, color='lime', alpha=0.15, linewidth=1)
    # if legend_loc:
    #     ax.legend(loc=legend_loc, framealpha=0.6)
    # else:
    #     ax.legend()
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, linewidth=0.6)
    return ax


def plot_merged_graph_3(ax, x1, y1, y1_smooth, label_1, x2, y2, y2_smooth, label_2, x3, y3, y3_smooth, label_3, title,
                        y_label, ylim=[], legend_loc=[], x_label="Timesteps"):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(x1, y1_smooth, color='dodgerblue', label=label_1)
    ax.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    ax.plot(x2, y2_smooth, color='orangered', label=label_2)
    ax.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    ax.plot(x3, y3_smooth, color='purple', label=label_3)
    ax.plot(x3, y3, color='purple', alpha=0.15, linewidth=1)
    # if legend_loc:
    #     ax.legend(loc=legend_loc, framealpha=0.6)
    # else:
    #     ax.legend()
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
        x1, y1 = merge_runs_to_xy("monitors/reward_type/1000/soft/distance/ppo_base_soft_1000", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/reward_type/1000/aggressive/distance/ppo_base_aggressive_1000",
                                  variable_type)
        y2_smooth = smooth_curve(y2, 100)

        if variable_type == "r":
            plot_merged_graph(
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Learning curve using distance-based reward function",
                y_label="Rewards",
                legend_loc="upper left",
                ax=axes[i],
                # ylim=[-9000, 7600]
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
        x1, y1 = merge_runs_to_xy("monitors/reward_type/1000/soft/action/ppo_base_action_soft_1000", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/reward_type/1000/aggressive/action/ppo_base_action_aggressive_1000",
                                  variable_type)
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
        x1, y1 = merge_runs_to_xy("monitors/reward_type/1000/soft/wheel_speed"
                                  "/ppo_base_wheel_speed_soft_1000",
                                  variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/reward_type/1000/aggressive/wheel_speed"
                                  "/ppo_base_wheel_speed_aggressive_1000",
                                  variable_type)
        y2_smooth = smooth_curve(y2, 100)

        if variable_type == "r":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Learning curve using wheel speed-based reward function",
                y_label="Rewards",
                # ylim=[-200, 8000],
                legend_loc="upper left"
            )
        elif variable_type == "score":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode score using wheel speed-based reward function",
                y_label="Score",
                legend_loc="upper left"
            )
        elif variable_type == "l":
            plot_merged_graph(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Soft",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Aggressive",
                title="Episode length using wheel speed-based reward function",
                y_label="Episode length (in timesteps)",
                # ylim=[-100, 5000],
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
    x, y = merge_runs_to_xy("monitors/increasing_difficulty/action/ppo_base_action_diff_increasing_soft_1000", 'score')
    x, y2 = merge_runs_to_xy("monitors/reward_type/1000/soft/action/ppo_base_action_soft_1000", "score")
    x, y3 = merge_runs_to_xy("monitors/increasing_difficulty/distance_continuous/ppo_cont_diff_increasing_soft_1000",
                             'score')
    x, y4 = merge_runs_to_xy("monitors/cont_reward_type/1000/soft/distance/ppo_cont_soft_1000", "score")
    x, y5 = merge_runs_to_xy("monitors/increasing_difficulty/distance_discrete/ppo_base_diff_increasing_soft_1000",
                             'score')
    x, y6 = merge_runs_to_xy("monitors/reward_type/1000/soft/distance/ppo_base_soft_1000", "score")
    x, y7 = merge_runs_to_xy("monitors/increasing_difficulty/wheel_speed"
                             "/ppo_cont_wheel_speed_diff_increasing_aggressive_1000", 'score')
    x, y8 = merge_runs_to_xy("monitors/cont_reward_type/1000/aggressive/wheel_speed"
                             "/ppo_cont_wheel_speed_aggressive_1000", "score")
    reward_types = ["Action", "Action (original)", "Distance cont", "Distance cont (original)", "Distance discrete",
                    "Distance discrete (original)", "Wheel speed", "Wheel speed (original)"]
    data = [y, y2, y3, y4, y5, y6, y7, y8]
    plt.title("Episode score, original vs. increasing difficulty")
    plt.xlabel("Score")
    plot = plt.boxplot(x=data, labels=reward_types, flierprops=dict(alpha=0.5), vert=0, patch_artist=True)
    # fill with colors
    colors = ['pink', 'pink', 'lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightsalmon', 'lightsalmon']
    for patch, color in zip(plot['boxes'], colors):
        patch.set_facecolor(color)
    # plt.xlim(-10, 310)
    plt.tight_layout()
    return plt


# Graph for increasing difficulty experiments
def graph_increasing_difficulty():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed

    for i, variable_type in enumerate(['r', 'l', 'score']):
        x1, y1 = merge_runs_to_xy("monitors/increasing_difficulty/action/ppo_base_action_diff_increasing_soft_1000",
                                  variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/increasing_difficulty/distance_discrete/ppo_base_diff_increasing_soft_1000",
                                  variable_type)
        y2_smooth = smooth_curve(y2, 100)
        x3, y3 = merge_runs_to_xy("monitors/increasing_difficulty/distance_continuous"
                                  "/ppo_cont_diff_increasing_soft_1000", variable_type)
        y3_smooth = smooth_curve(y3, 100)
        x4, y4 = merge_runs_to_xy("monitors/increasing_difficulty/wheel_speed"
                                  "/ppo_cont_wheel_speed_diff_increasing_aggressive_1000", variable_type)
        y4_smooth = smooth_curve(y4, 100)

        if variable_type == "r":
            plot_merged_graph_4(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Action discrete (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Distance continuous (soft)",
                x4=x4, y4=y4, y4_smooth=y4_smooth, label_4="Wheel speed continuous (aggressive)",
                title="Learning curve of different reward functions",
                y_label="Rewards",
                ylim=[-5000, 20000],
                legend_loc="lower right"
            )
        elif variable_type == "score":
            plot_merged_graph_4(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Action discrete (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Distance continuous (soft)",
                x4=x4, y4=y4, y4_smooth=y4_smooth, label_4="Wheel speed continuous (aggressive)",
                title="Episode score of different reward functions",
                y_label="Score",
                legend_loc="lower right"
            )
        elif variable_type == "l":
            plot_merged_graph_4(
                ax=axes[i],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Action discrete (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Distance continuous (soft)",
                x4=x4, y4=y4, y4_smooth=y4_smooth, label_4="Wheel speed continuous (aggressive)",
                title="Episode length of different reward functions",
                y_label="Episode length (in timesteps)",
                # ylim=[-100, 5000],
                legend_loc="upper left"
            )
    fig.tight_layout(pad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2)
    fig.subplots_adjust(bottom=0.2)


def airtime_comparison():
    # Distance-based reward function
    x1, y1 = merge_runs_to_xy("monitors/airtime/distance_continuous/ppo_cont_airtime_soft_1000", 'total_airtime')
    y1_smooth = smooth_curve(y1, 100)
    x2, y2 = merge_runs_to_xy("monitors/airtime/distance_discrete/ppo_base_airtime_soft_1000", 'total_airtime')
    y2_smooth = smooth_curve(y2, 100)
    x3, y3 = merge_runs_to_xy("monitors/airtime/wheel_speed/ppo_cont_wheel_speed_airtime_aggressive_1000",
                              'total_airtime')
    y3_smooth = smooth_curve(y3, 100)

    plt.title("Airtime curve of different reward functions")
    plt.ylabel("Airtime")
    plt.xlabel("Timesteps")

    plt.plot(x1, y1_smooth, color='dodgerblue', label="Distance continuous (soft)")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Distance discrete (soft)")
    plt.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    plt.plot(x3, y3_smooth, color='darkviolet', label="Wheel speed (aggressive)")
    plt.plot(x3, y3, color='darkviolet', alpha=0.15, linewidth=1)
    plt.legend(framealpha=0.6, loc='upper left')
    plt.grid(True, linewidth=0.6)
    return plt


def graph_airtime():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Adjust figsize as needed

    for i, variable_type in enumerate(['r', 'l', 'score', 'total_airtime']):
        x1, y1 = merge_runs_to_xy("monitors/airtime/distance_continuous/ppo_cont_airtime_soft_1000", variable_type)
        y1_smooth = smooth_curve(y1, 100)
        x2, y2 = merge_runs_to_xy("monitors/airtime/distance_discrete/ppo_base_airtime_soft_1000", variable_type)
        y2_smooth = smooth_curve(y2, 100)
        x3, y3 = merge_runs_to_xy("monitors/airtime/wheel_speed/ppo_cont_wheel_speed_airtime_aggressive_1000",
                                  variable_type)
        y3_smooth = smooth_curve(y3, 100)

        if variable_type == "r":
            plot_merged_graph_3(
                ax=axes[0, 0],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Distance continuous (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Wheel speed (aggressive)",
                title="Learning curve of different reward functions",
                y_label="Rewards",
                # ylim=[-5000, 20000],
                legend_loc="lower right"
            )
        elif variable_type == "score":
            plot_merged_graph_3(
                ax=axes[0, 1],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Distance continuous (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Wheel speed (aggressive)",
                title="Episode score of different reward functions",
                y_label="Score",
                legend_loc="lower right"
            )
        elif variable_type == "l":
            plot_merged_graph_3(
                ax=axes[1, 0],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Distance continuous (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Wheel speed (aggressive)",
                title="Episode length of different reward functions",
                y_label="Episode length (in timesteps)",
                # ylim=[-100, 5000],
                legend_loc="upper left"
            )
        elif variable_type == "total_airtime":
            plot_merged_graph_3(
                ax=axes[1, 1],
                x1=x1, y1=y1, y1_smooth=y1_smooth, label_1="Distance continuous (soft)",
                x2=x2, y2=y2, y2_smooth=y2_smooth, label_2="Distance discrete (soft)",
                x3=x3, y3=y3, y3_smooth=y3_smooth, label_3="Wheel speed (aggressive)",
                title="Airtime of different reward functions",
                y_label="Episode length (in timesteps)",
                # ylim=[-100, 5000],
                legend_loc="upper left"
            )
    fig.tight_layout(pad=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2)
    fig.subplots_adjust(bottom=0.1)


def position_time_comparison():
    plt.figure(figsize=(10, 8))
    # Distance-based reward function
    y1, x1 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_discrete_action_soft_1000",
                                  variable_type="position_list")
    y1_smooth = smooth_curve(y1, 500)
    y2, x2 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_wheel_speed_soft_1000",
                                  variable_type="position_list")
    y2_smooth = smooth_curve(y2, 500)
    y3, x3 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_wheel_speed_aggressive_1000",
                                  variable_type="position_list")
    y3_smooth = smooth_curve(y3, 500)
    y4, x4 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_distance_soft_1000",
                                  variable_type="position_list")
    y4_smooth = smooth_curve(y4, 500)
    y5, x5 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_base_distance_soft_1000",
                                  variable_type="position_list")
    y5_smooth = smooth_curve(y5, 500)

    plt.title("Position of agent during episode evaluation")
    plt.xlabel("Timesteps")
    plt.ylabel("Position of agent")

    plt.plot(x1, y1_smooth, color='dodgerblue', label="Action discrete soft")
    plt.plot(x1, y1, color='dodgerblue', alpha=0.15, linewidth=1)
    plt.plot(x2, y2_smooth, color='orangered', label="Wheel speed continuous soft")
    plt.plot(x2, y2, color='orangered', alpha=0.15, linewidth=1)
    plt.plot(x3, y3_smooth, color='purple', label="Wheel speed continuous aggressive")
    plt.plot(x3, y3, color='purple', alpha=0.15, linewidth=1)
    plt.plot(x4, y4_smooth, color='lime', label="Distance continuous soft")
    plt.plot(x4, y4, color='lime', alpha=0.15, linewidth=1)
    plt.plot(x5, y5_smooth, color='yellow', label="Distance discrete soft")
    plt.plot(x5, y5, color='yellow', alpha=0.15, linewidth=1)
    plt.grid(True, linewidth=0.5)
    plt.legend(framealpha=0.6, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncols=2)
    plt.tight_layout()
    # plt.ylim(-6, 1000)
    return plt


def eval_boxplot_score():
    # Distance-based reward function
    y1, x1 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_discrete_action_soft_1000",
                                  variable_type="score")
    y2, x2 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_wheel_speed_soft_1000",
                                  variable_type="score")
    y3, x3 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_wheel_speed_aggressive_1000",
                                  variable_type="score")
    y4, x4 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_cont_distance_soft_1000",
                                  variable_type="score")
    y5, x5 = merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_base_distance_soft_1000",
                                  variable_type="score")

    reward_types = ["Discrete action soft, Continuous wheel speed soft, co"]
    data = [y1, y2, y3, y4, y5]
    plt.title("Episode score, original vs. increasing difficulty")
    plt.xlabel("Score")
    plot = plt.boxplot(x=data, labels=reward_types, flierprops=dict(alpha=0.5), vert=0, patch_artist=True)
    # fill with colors
    colors = ['pink', 'pink', 'lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightsalmon', 'lightsalmon']
    for patch, color in zip(plot['boxes'], colors):
        patch.set_facecolor(color)
    # plt.xlim(-10, 310)
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # graph_rewards_wheel_speed()
    # plt.savefig("1000_cont_wheel_speed_merged", dpi=300)
    # graph_rewards_distance()
    # plt.savefig("1000_cont_distance_merged", dpi=300)
    # graph_airtime()
    # merge_runs_dir_to_xy(path="monitors/evaluation/eval_ppo_discrete_action_soft_1000",
    #                      variable_type="position_list")
    # plt.show()
    eval_boxplot_score()
    plt.savefig("boxplot_score_evaluations", dpi=300)
    plt.show()

    # df = pd.read_csv("ppo_cont_wheel_speed_300_2.monitor.csv", header=1)
    # print(df)
    # data = load_results("monitors/reward_type/300/aggressive/distance/0")
    # print(data)
