from state_detection.wave_pipeline.waves import add_timeseries_features, add_summary_features, add_summary_comparison_features, summarize_runs
def run_wave_pipeline(
    df,
    signals,
    group,
    base_signal=None,
    smooth_window=20,
    diff_lag=10,
    eps=0,
):
    df = df.copy()
    base_signal = base_signal or signals[0]

    # 1. Add timeseries features for all signals
    for signal in signals:
        df = add_timeseries_features(
            df,
            signal=signal,
            group=group,
            smooth_window=smooth_window,
            diff_lag=diff_lag,
            eps=eps,
        )

    # 2. Build base wave summary
    summarydf = add_summary_features(
        df,
        signal=base_signal,
        group=group,
    )

    summarydf = add_summary_comparison_features(
        summarydf,
        signal=base_signal,
        group=group,
    )

    base_wave_col = f"{base_signal}_wave_num"

    # 3. Add summaries for related signals
    for signal in signals:
        if signal == base_signal:
            continue

        s = add_summary_features(
            df,
            signal=signal,
            group=group,
        )

        s = add_summary_comparison_features(
            s,
            signal=signal,
            group=group,
        )

        s = s.reset_index(drop=False)

        keep_cols = group + [
            col for col in s.columns
            if col.startswith(f"{signal}_")
        ]

        s = s[keep_cols]

        summarydf = (
            summarydf.reset_index(drop=False)
            .merge(s, on=group, how="left")
            .set_index(group + [base_wave_col])
        )

    summarydf = summarydf.reset_index(drop=False)

    # 4. Aggregate wave summaries to run-level summary
    run_summary = None

    for signal in signals:
        s = summarize_runs(
            summarydf,
            signal=signal,
            group=group,
        )

        if run_summary is None:
            run_summary = s
        else:
            run_summary = run_summary.join(s)

    run_summary = run_summary.reset_index()

    return run_summary
