from state_detection.operators.events import enter_state, exit_state, event_number
from state_detection.operators.signals import smooth

def add_timeseries_features(df, signal, group, smooth_window=20, diff_lag=10, eps=0):
    df = df.copy()

    smooth_col = f"{signal}_smooth"
    rising_col = f"{signal}_rising"
    falling_col = f"{signal}_falling"
    enter_col = f"enter_{rising_col}"
    exit_col = f"exit_{rising_col}"
    wave_col = f"{signal}_wave_num"

    df[smooth_col] = smooth(df, signal, group, smooth_window)

    delta = (
        df.groupby(group)[smooth_col]
          .transform(lambda s: s.diff(diff_lag))
    )

    df[rising_col] = delta > eps
    df[falling_col] = delta < -eps

    df[enter_col] = enter_state(df[rising_col])
    df[exit_col] = exit_state(df[rising_col])

    df[wave_col] = event_number(df, enter_col, group)

    return df

def add_summary_features(df, signal, group):
    group_cols = group if isinstance(group, list) else [group]

    wave_col = f"{signal}_wave_num"
    rising_col = f"{signal}_rising"
    falling_col = f"{signal}_falling"
    smooth_col = f"{signal}_smooth"

    summarydf = (
        df.groupby(group_cols + [wave_col])
          .agg(
              **{
                  f"{signal}_rise_time": (rising_col, "sum"),
                  f"{signal}_fall_time": (falling_col, "sum"),
                  f"{signal}_peak": (smooth_col, "max"),
                  f"{signal}_trough": (smooth_col, "min"),
                  f"{signal}_peak_index": (
                      "sample",
                      lambda s: s.iloc[df.loc[s.index, smooth_col].argmax()]
                  ),
              }
          )
    )

    summarydf[f"{signal}_amplitude"] = (
        summarydf[f"{signal}_peak"] - summarydf[f"{signal}_trough"]
    ) / 2

    summarydf[f"{signal}_wave_duration"] = (
        summarydf[f"{signal}_rise_time"] +
        summarydf[f"{signal}_fall_time"]
    )

    summarydf = summarydf[
        (summarydf.index.get_level_values(wave_col) > 0) &
        (summarydf[f"{signal}_wave_duration"] >= 5)
    ]

    return summarydf

def add_summary_comparison_features(summarydf, signal, group):
    summarydf = summarydf.copy()

    group_cols = group if isinstance(group, list) else [group]

    amp = f"{signal}_amplitude"
    rise = f"{signal}_rise_time"
    fall = f"{signal}_fall_time"

    if f"{signal}_peak_index" in summarydf.columns:
        summarydf[f"{signal}_period"] = (
            summarydf.groupby(group_cols)[f"{signal}_peak_index"]
                 .diff()
        )

    summarydf[f"{signal}_amplitude_prev"] = (
        summarydf.groupby(group_cols)[amp].shift(1)
    )

    summarydf[f"{signal}_damping_ratio"] = (
        summarydf[amp] / summarydf[f"{signal}_amplitude_prev"]
    )

    summarydf[f"{signal}_decay_rate"] = (
        1 - summarydf[f"{signal}_damping_ratio"]
    )

    summarydf[f"{signal}_symmetry"] = (
        summarydf[rise] / (summarydf[rise] + summarydf[fall])
    )

    return summarydf

def summarize_runs(summarydf, signal, group):
    return (
        summarydf.groupby(group)
        .agg(
            **{
                f"{signal}_amplitude_mean": (f"{signal}_amplitude", "mean"),
                f"{signal}_amplitude_std": (f"{signal}_amplitude", "std"),

                f"{signal}_period_mean": (f"{signal}_period", "mean"),
                f"{signal}_period_std": (f"{signal}_period", "std"),

                f"{signal}_wave_duration_mean": (f"{signal}_wave_duration", "mean"),
                f"{signal}_wave_duration_std": (f"{signal}_wave_duration", "std"),

                f"{signal}_rise_time_mean": (f"{signal}_rise_time", "mean"),
                f"{signal}_fall_time_mean": (f"{signal}_fall_time", "mean"),

                f"{signal}_symmetry_mean": (f"{signal}_symmetry", "mean"),
                f"{signal}_symmetry_std": (f"{signal}_symmetry", "std"),

                f"{signal}_damping_ratio_mean": (f"{signal}_damping_ratio", "mean"),
                f"{signal}_damping_ratio_std": (f"{signal}_damping_ratio", "std"),

                f"{signal}_decay_rate_mean": (f"{signal}_decay_rate", "mean"),
                f"{signal}_decay_rate_std": (f"{signal}_decay_rate", "std"),

                # Number of detected oscillations
                f"{signal}_wave_count": (f"{signal}_wave_num", "count"),

                # Largest oscillation
                f"{signal}_max_amplitude": (f"{signal}_amplitude", "max"),

                # Longest oscillation
                f"{signal}_max_duration": (f"{signal}_wave_duration", "max"),

                # Fraction of waves that are growing
                f"{signal}_fraction_growing": (
                    f"{signal}_damping_ratio",
                    lambda x: (x > 1).mean()
                ),

                # Fraction of waves that are damping
                f"{signal}_fraction_damping": (
                    f"{signal}_damping_ratio",
                    lambda x: (x < 1).mean()
                ),
            }
        )
    )
