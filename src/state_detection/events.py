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


def event_number(df, enter_col, group=None):
    if group is None:
        return df[enter_col].cumsum()
    return df.groupby(group)[enter_col].cumsum()
