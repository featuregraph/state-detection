from state_detection.state_operators import smooth, rising_state, falling_state, enter_state, exit_state, event_number
def add_wave_features(
    df,
    signal,
    group="simulation_run",
    smooth_window=20,
    diff_lag=10,
    eps=0,
):
    smooth_col = f"{signal}_smooth"
    rising_col = f"{signal}_rising"
    falling_col = f"{signal}_falling"
    enter_col = f"enter_{rising_col}"
    exit_col = f"exit_{rising_col}"
    wave_col = f"{signal}_wave_num"

    df[smooth_col] = smooth(df, signal, group, smooth_window)

    df[rising_col] = (
        df.groupby(group)[smooth_col]
          .transform(lambda s: rising_state(s, lag=diff_lag, eps=eps))
    )

    df[falling_col] = (
        df.groupby(group)[smooth_col]
          .transform(lambda s: falling_state(s, lag=diff_lag, eps=eps))
    )

    df[enter_col] = enter_state(df[rising_col], df[group])
    df[exit_col] = exit_state(df[rising_col], df[group])
    df[wave_col] = event_number(df[enter_col], df[group])

    return df

def measure_wave(df, wave_col, signals, group="simulation_run"):
    agg_spec = {}

    for signal in signals:
        agg_spec[f"{signal}_peak"] = (signal, "max")
        agg_spec[f"{signal}_trough"] = (signal, "min")
        agg_spec[f"{signal}_mean"] = (signal, "mean")

    waves = (
        df.groupby([group, wave_col])
          .agg(**agg_spec)
          .reset_index()
    )

    for signal in signals:
        waves[f"{signal}_peak_to_trough"] = (
            waves[f"{signal}_peak"] - waves[f"{signal}_trough"]
        )
        waves[f"{signal}_amplitude"] = waves[f"{signal}_peak_to_trough"] / 2

    return waves