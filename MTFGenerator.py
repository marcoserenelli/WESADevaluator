import math

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pyts.image import MarkovTransitionField
from tensorflow import keras


class MTFGenerator(keras.utils.Sequence):

    def __init__(self,
                 subjects,
                 all_subject_data,
                 features,
                 labels,
                 batch_size=64,
                 image_size=128,
                 n_bins=5,
                 window_size=60 * 35,
                 time_step=35,
                 shuffle=True,
                 seed=None):

        if seed is not None:
            np.random.seed(seed)
        # Load the specified subject data
        self.all_subject_data = all_subject_data
        self.features = features
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_bins = n_bins
        self.gaf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
        # Create a list of possible windows that will be converted into images
        self.idx_list = list()

        for subject in subjects:
            for label in labels:
                curr_idx_interval = all_subject_data[subject].index[
                    all_subject_data[subject]['label'] == label].tolist()
                for i in range(curr_idx_interval[0], curr_idx_interval[-1] - window_size + 1, time_step):
                    self.idx_list.append((subject, i, label))

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx_list)

    def get_labels(self):
        return np.array([y for _, _, y in self.idx_list])

    def get_image(self, subject, i):
        current_window = self.all_subject_data[subject].loc[i:i + self.window_size - 1, self.features].to_numpy().T
        # Apply KBinsDiscretizer
        kbins = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform', subsample=None)
        current_window = kbins.fit_transform(current_window)
        return np.transpose(self.gaf.transform(current_window), (1, 2, 0))

    def __len__(self):
        return math.ceil(len(self.idx_list) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.idx_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([self.get_image(subject, i) for subject, i, _ in batch]), np.array([y for _, _, y in batch])
