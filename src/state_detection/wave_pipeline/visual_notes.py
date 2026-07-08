import pandas as pd
from state_detection.data_formatting.rename_map import rename_map
from state_detection.wave_pipeline.wave_pipeline import run_wave_pipeline

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