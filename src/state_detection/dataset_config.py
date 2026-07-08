class DatasetConfig:

    def __init__(self, name, time_col, group_cols, signals, sampling_rate):
        self.name = name
        self.time_col = time_col
        self.group_cols = group_cols
        self.signals = signals
        self.sampling_rate = sampling_rate

class SignalConfig:

    def __init__(self, signal):
        smooth_col = f"{signal}_smooth"
        rising_col = f"{signal}_rising"
        falling_col = f"{signal}_falling"
        enter_rising_col = f"enter_{rising_col}"
        exit_rising_col = f"exit_{rising_col}"


