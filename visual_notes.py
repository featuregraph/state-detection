import numpy as np
import pandas as pd
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from lib.common.plotting.timeseries import plot_horizontal as plot
from state_detection.rename_map import rename_map
from state_detection.wave_pipeline import run_wave_pipeline

df = pd.read_csv("TEP_Faulty_Training.csv")
df = df.rename(columns=rename_map)

run_summary = run_wave_pipeline(
    df,
    signals=["reactor_pressure", "reactor_temperature"],
    group=["fault_number", "simulation_run"],
    base_signal="reactor_pressure",
    smooth_window=20,
    diff_lag=10,
    eps=0,
)

run_summary.to_csv('wave_summary_simulation_run_faults.csv')