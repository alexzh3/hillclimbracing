from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Callable, List, Optional, Tuple
import pandas as pd

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def smooth_curve(y, window, poly=1):
    return savgol_filter(y, window, poly)


def transfer_to_xy(data_frame: pd.DataFrame, x_axis: str, y_var_values) -> Tuple[np.ndarray, np.ndarray]:
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = y_var_values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = y_var_values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = y_var_values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_rewards(title, df, y_var_values, y_label):
    x, y = transfer_to_xy(df, "timesteps", y_var_values)  # for rewards
    y = smooth_curve(y, window=50, poly=1)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel(y_label)
    plt.title(title)
    # plt.xlim(0, 1e6)
    plt.show()


if __name__ == "__main__":
    log_dir = "monitors/base"
    data = load_results(log_dir)
    plot_rewards(title="Learning curve of baseline method", df=data, y_var_values=data.r.values, y_label="Smoothed rewards")
