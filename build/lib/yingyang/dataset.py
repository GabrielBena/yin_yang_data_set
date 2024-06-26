import numpy as np
from torch.utils.data.dataset import Dataset


class YinYangDataset(Dataset):
    def __init__(
        self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None, rotation=None
    ):
        super().__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
        self.data = []
        self.targets = []
        self.class_names = ["yin", "yang", "dot"]
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1.0 - x
            y_flipped = 1.0 - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.data.append(val)
            self.targets.append(c)
        self.data, self.targets = np.array(self.data), np.array(self.targets)
        if rotation is not None:
            rotation_matrix = np.array(
                [
                    [np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)],
                ]
            )
            self.data -= 0.5  ## center the data

            self.data[:, :2] = np.dot(self.data[:, :2], rotation_matrix)
            self.data[:, 2:] = np.dot(self.data[:, 2:], rotation_matrix)
            self.data += 0.5  ## replace between 0 and 1

            # self.data = (self.data - self.data.min()) / (
            #     self.data.max() - self.data.min()
            # )
            # normalize_transform =
            # if self.transform is None :

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2.0 * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def __getitem__(self, index):
        sample = (self.data[index].copy(), self.targets[index])
        if self.transform:
            if sample[0].ndim == 1:
                sample = (sample[0][None, :], sample[1])
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.targets)

