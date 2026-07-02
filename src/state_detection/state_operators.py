def smooth(df, signal, group, window):
    return (
        df.groupby(group)[signal]
          .rolling(window, min_periods=window)
          .mean()
          .reset_index(level=0, drop=True)
    )


def rising_state(series, lag=10, eps=0):
    return series.diff(lag).gt(eps)


def falling_state(series, lag=10, eps=0):
    return series.diff(lag).lt(-eps)


def enter_state(state, group=None):
    x = state.astype(int)
    if group is None:
        return x.diff().eq(1)
    return x.groupby(group).diff().eq(1)


def exit_state(state, group=None):
    x = state.astype(int)
    if group is None:
        return x.diff().eq(-1)
    return x.groupby(group).diff().eq(-1)


def event_number(enter, group=None):
    if group is None:
        return enter.cumsum()
    return enter.groupby(group).cumsum()
