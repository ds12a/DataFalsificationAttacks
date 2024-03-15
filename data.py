from torch.utils.data import Dataset
import numpy as np


# Fixed windows
class FixedWindowSelector(Dataset):
    def __init__(self, data_frame, q):
        self.data = data_frame.values
        self.q = q

    def __len__(self):
        return self.data.shape[0] // self.q

    def __getitem__(self, index):
        A = self.data[index * self.q : (index + 1) * self.q]
        return (A - A.min()) / (A.max() if A.max() > 0 else 1)


# All windows
class WindowSelector(Dataset):
    def __init__(self, data_frame, window):
        self.data = data_frame.values
        self.window = window

    def __len__(self):
        return self.data.shape[0] - self.window + 1

    def __getitem__(self, index):
        A = self.data[index : index + self.window]
        return A


class ChanneledWindowSelector(Dataset):
    def __init__(self, data_frames, window):
        self.data = data_frames
        self.window = window

    def __len__(self):
        return self.data.shape[0] - self.window + 1

    def __getitem__(self, index):
        A = self.data[:][index : index + self.window]
        return A


class NormalizedWindowSelector(Dataset):
    def __init__(self, data_frame, window, max_value):
        self.data = data_frame.values
        self.window = window
        self.max_value = max_value

    def __len__(self):
        return self.data.shape[0] - self.window + 1

    def __getitem__(self, index):
        A = self.data[index : index + self.window]
        return A / self.max_value


class MultiClassNormalizedWindowSelector(Dataset):
    def __init__(self, data_frames, window, max_value):
        self.data = data_frames
        self.window = window
        self.max_value = max_value

    def __len__(self):
        return self.data.size - (self.window - 1) * self.data.shape[0]

    def __getitem__(self, index):
        set_index = 0
        while (set_index + 1) < self.data.shape[0] and index - (
            self.data[set_index].shape[0] - self.window
        ) > 0:
            index -= self.data[set_index].shape[0] - self.window + 1
            set_index += 1

        A = self.data[set_index][index : index + self.window]
        return (A / self.max_value, np.array(set_index, dtype=np.float64))
