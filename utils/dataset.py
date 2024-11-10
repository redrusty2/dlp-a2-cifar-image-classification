import torch
import numpy as np
import pickle
from typing import NamedTuple
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Calculated mean and standard deviation of the CIFAR-10 train set
mean = [0.49140089750289917, 0.48215895891189575, 0.4465307891368866]
std = [0.2470327913761139, 0.243484228849411, 0.261587530374527]

# Copied from: https://www.kaggle.com/code/ayushnitb/cifar10-custom-resnet-cnn-pytorch-97-acc
augment_compose = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1, inplace=False),
    ]
)

augment_compose2 = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False),
    ]
)

base_compose = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class CF10AugmentedDataset(Dataset):
    raw_imgs: NDArray[np.uint8]
    raw_labels: NDArray[np.uint8]
    imgs: list[torch.Tensor]
    labels: list[torch.Tensor]

    def __init__(
        self, raw_imgs: NDArray[np.uint8], raw_labels: NDArray[np.uint8], n_augments=2
    ) -> None:
        self.raw_imgs = raw_imgs
        self.raw_labels = raw_labels
        self.imgs, self.labels = transform_train_set(raw_imgs, raw_labels, n_augments)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        return img, label


class CF10FlyAugmentedDataset(Dataset):
    raw_imgs: NDArray[np.uint8]
    raw_labels: NDArray[np.uint8]
    imgs: list[Image.Image]
    labels: NDArray[np.uint8]

    def __init__(
        self, raw_imgs: NDArray[np.uint8], raw_labels: NDArray[np.uint8]
    ) -> None:
        self.raw_imgs = raw_imgs
        self.labels = raw_labels
        self.imgs = []
        for img, label in zip(raw_imgs, raw_labels):
            self.imgs.append(Image.fromarray((img).astype(np.uint8)))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs[idx].copy()
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)

        # if np.random.rand() > 0.5:
        img = augment_compose2(img)
        # else:
        #     img = base_compose(img)

        return img, label


class CF10Dataset(Dataset):
    raw_imgs: NDArray[np.uint8]
    raw_labels: NDArray[np.uint8]
    imgs: list[torch.Tensor]
    labels: list[torch.Tensor]

    def __init__(
        self, raw_imgs: NDArray[np.uint8], raw_labels: NDArray[np.uint8]
    ) -> None:
        self.raw_imgs = raw_imgs
        self.raw_labels = raw_labels
        self.imgs, self.labels = transform_test_set(raw_imgs, raw_labels)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.uint8)
        return img, label


class CFSet(NamedTuple):
    imgs: NDArray[np.uint8]
    labels: NDArray[np.uint8]


def load_train_set(base_path: str) -> CFSet:
    train_batches = []
    for i in range(1, 6):
        with open(f"{base_path}/data_batch_{i}", "rb") as f:
            train_batches.append(pickle.load(f, encoding="bytes"))

    train_imgs = (
        np.concatenate([batch[b"data"] for batch in train_batches], dtype=np.uint8)
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
    )

    train_labels = np.concatenate(
        [batch[b"labels"] for batch in train_batches], dtype=np.int8
    ).astype(np.uint8)

    return CFSet(train_imgs, train_labels)


def load_test_set(base_path: str) -> CFSet:
    with open(f"{base_path}/test_batch", "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")

    test_imgs: NDArray[np.uint8] = (
        test_batch[b"data"]
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
        .astype(np.uint8)
    )

    test_labels: NDArray[np.uint8] = np.array(
        test_batch[b"labels"], dtype=np.int8
    ).astype(np.uint8)

    return CFSet(test_imgs, test_labels)


def load_labels_map(base_path: str) -> dict[int, str]:
    with open(f"{base_path}/batches.meta", "rb") as f:
        label_names = pickle.load(f)["label_names"]

    labels_map = {i: label for i, label in enumerate(label_names)}

    return labels_map


def transform_train_set(
    raw_train_imgs, raw_train_labels, n_augments
) -> tuple[list, list]:
    labels = []
    imgs = []
    for img, label in zip(raw_train_imgs, raw_train_labels):
        # Append the original image
        imgs.append(base_compose(Image.fromarray((img).astype(np.uint8))))
        labels.append(label)
        # Create augmented images for each image
        for i in range(n_augments):
            imgs.append(augment_compose2(Image.fromarray((img).astype(np.uint8))))
            labels.append(label)

    return imgs, labels


def transform_test_set(raw_test_imgs, raw_test_labels) -> tuple[list, list]:
    labels = []
    imgs = []
    for img, label in zip(raw_test_imgs, raw_test_labels):
        imgs.append(base_compose(Image.fromarray((img).astype(np.uint8))))
        labels.append(label)

    return imgs, labels


def create_train_validation_split(
    data: CFSet, val_split: float = 0.1
) -> tuple[CFSet, CFSet]:
    classes = np.unique(data.labels)

    train_idxs = []
    val_idxs = []

    for c in classes:
        class_idxs = np.where(data.labels == c)[0]
        np.random.shuffle(class_idxs)
        split = int(np.floor(val_split * len(class_idxs)))
        train_idxs.extend(class_idxs[split:])
        val_idxs.extend(class_idxs[:split])

    train_imgs = data.imgs[train_idxs]
    train_labels = data.labels[train_idxs]
    val_imgs = data.imgs[val_idxs]
    val_labels = data.labels[val_idxs]

    return CFSet(train_imgs, train_labels), CFSet(val_imgs, val_labels)


# From https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y, alpha=1.0):
    """Apply Mixup on a batch of inputs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Custom loss function for Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
