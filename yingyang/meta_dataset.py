from yingyang.dataset import YinYangDataset
from torchmeta.utils.data import ClassDataset, CombinationMetaDataset
import numpy as np
from torch.utils.data import ConcatDataset


def get_seeds_and_rotations(n_tasks):

    train_rotations = np.linspace(0, np.pi / 2, n_tasks, endpoint=False)
    test_rotations = np.linspace(np.pi / 2, np.pi / 2, n_tasks, endpoint=False)
    val_rotations = np.linspace(np.pi / 2, np.pi, n_tasks, endpoint=False)

    train_seeds = np.arange(0, 2 * n_tasks, 2)
    test_seeds = np.arange(2 * n_tasks, 4 * n_tasks, 2)
    val_seeds = np.arange(4 * n_tasks, 6 * n_tasks, 2)

    rotations = {"train": train_rotations, "test": test_rotations, "val": val_rotations}
    seeds = {"train": train_seeds, "test": test_seeds, "val": val_seeds}

    return rotations, seeds


class YinYangClassDataset(ClassDataset):
    def __init__(
        self,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        target_transform=None,
        data_config=None,
    ):

        if meta_train:
            split_name = "train"
        if meta_val:
            split_name = "val"
        if meta_test:
            split_name = "test"
        self.split_name = split_name

        self.transform = transform
        self.target_transform = target_transform

        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
        )

        self.data_config = data_config
        self.rotations, self.seeds = get_seeds_and_rotations(data_config["n_tasks_per_split"])
        self._labels = np.arange(len(self.rotations[split_name]))
        self._num_classes = len(self.rotations[split_name])
        self.loaded_datasets = {}

    @property
    def labels(self):
        return self._labels

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        rotation = self.rotations[self.split_name][index]
        seed = self.seeds[self.split_name][index]

        if rotation not in self.loaded_datasets:
            trainset = YinYangDataset(
                size=self.data_config["dataset_size"],
                seed=seed,
                rotation=rotation,
                transform=self.transform,
            )
            testset = YinYangDataset(
                size=self.data_config["dataset_size"],
                seed=seed + 1,
                rotation=rotation,
                transform=self.transform,
            )
            self.loaded_datasets[rotation] = ConcatDataset([trainset, testset])
            if self.transform is not None:
                self.loaded_datasets[rotation].transform = self.transform
            self.loaded_datasets[rotation].index = index
            self.loaded_datasets[rotation].target_transform_append = lambda x: None

        return self.loaded_datasets[rotation]


class YingYangMetaDataset(CombinationMetaDataset):
    def __init__(
        self,
        num_classes_per_task=None,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        data_config=None,
        dataset_transform=None,
    ):
        if meta_train:
            split_name = "train"
        if meta_val:
            split_name = "val"
        if meta_test:
            split_name = "test"
        self.split_name = split_name

        dataset = YinYangClassDataset(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            transform=transform,
            data_config=data_config,
        )
        super().__init__(
            dataset,
            num_classes_per_task=num_classes_per_task,
            dataset_transform=dataset_transform,
        )
