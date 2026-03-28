import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from utils import MultiRandomChoice
from typing import Optional


class MVTecDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        class_name: str,
        phase: str = "test",
        resize: int = 256,
        cropsize: int = 224,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        anomaly_generators: list[v2.Transform] | None = None,
    ) -> None:
        self.resize = resize
        self.cropsize = cropsize
        self.phase = phase

        if anomaly_generators is not None:
            self.augmenter = MultiRandomChoice(
                transforms=anomaly_generators,
                probabilities=None,
                num_transforms=1,
                fixed_num_transforms=True,
            )
        else:
            self.augmenter = None

        img_dir = os.path.join(dataset_path, class_name, phase)
        gt_dir = os.path.join(dataset_path, class_name, "ground_truth")
        self.x: list[Path] = []
        self.y: list[int] = []
        self.mask_paths: list[Optional[Path]] = []

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            fpaths = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            self.x.extend(fpaths)
            if img_type == "good":
                self.y.extend([0] * len(fpaths))
                self.mask_paths.extend([None] * len(fpaths))
            else:
                self.y.extend([1] * len(fpaths))
                gt_type_dir = os.path.join(gt_dir, img_type)
                for f in fpaths:
                    fname = os.path.splitext(os.path.basename(f))[0]
                    self.mask_paths.append(
                        os.path.join(gt_type_dir, f"{fname}_mask.png")
                    )

        self.transform_x = v2.Compose(
            [
                v2.Resize(
                    size=resize,
                    interpolation=v2.InterpolationMode.LANCZOS,
                ),
                v2.CenterCrop(cropsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.normalize = v2.Normalize(mean=mean, std=std)

        self.transform_m = v2.Compose(
            [
                v2.Resize(
                    size=resize,
                    interpolation=v2.InterpolationMode.NEAREST,
                ),
                v2.CenterCrop(cropsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | int, torch.Tensor, str]:
        image = self.transform_x(Image.open(self.x[idx]).convert("RGB"))
        if self.phase == "train":
            if self.augmenter is not None:
                augmented_image, mask = self.augmenter(image)
            else:
                augmented_image = image.clone().detach()
                mask = torch.zeros([1, self.cropsize, self.cropsize])

            augmented_image = self.normalize(augmented_image)
            image = self.normalize(image)
            return image, augmented_image, mask, self.x[idx]
        else:
            image = self.normalize(image)
            mask = (
                torch.zeros([1, self.cropsize, self.cropsize])
                if self.y[idx] == 0
                else self.transform_m(Image.open(self.mask_paths[idx]).convert("L"))
            )
        return image, self.y[idx], mask, self.x[idx]
