from state_detection.states import rising_state, falling_state
from state_detection.events import enter_state, exit_state, event_number
from state_detection.signals import smooth

def add_wave_features(df, signal, group, smooth_window=20, diff_lag=10, eps=0):
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

def add_wave_comparison_features(waves, group, signal):
    waves = waves.copy()

    group_cols = group if isinstance(group, list) else [group]

    amp = f"{signal}_amplitude"
    rise = f"{signal}_rise_time"
    fall = f"{signal}_fall_time"

    if f"{signal}_peak_index" in waves.columns:
        waves[f"{signal}_period"] = (
            waves.groupby(group_cols)[f"{signal}_peak_index"]
                 .diff()
        )

    waves[f"{signal}_amplitude_prev"] = (
        waves.groupby(group_cols)[amp].shift(1)
    )

    waves[f"{signal}_damping_ratio"] = (
        waves[amp] / waves[f"{signal}_amplitude_prev"]
    )

    waves[f"{signal}_decay_rate"] = (
        1 - waves[f"{signal}_damping_ratio"]
    )

    waves[f"{signal}_symmetry"] = (
        waves[rise] / (waves[rise] + waves[fall])
    )

    return waves