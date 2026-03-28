import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Literal


def crop_and_patch(
    image: torch.Tensor,
    patch_width: int,
    patch_height: int,
    image_transform: v2.Transform,
    mode: Literal["normal", "scar"],
    device: torch.device,
    rotation_transform: v2.Transform | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_height, original_width = image.shape[1:3]
    patch_left, new_left = torch.randint(
        0, original_width - patch_width + 1, (2,), device=device
    ).tolist()
    patch_top, new_top = torch.randint(
        0, original_height - patch_height + 1, (2,), device=device
    ).tolist()

    patch_right = patch_left + patch_width
    patch_bottom = patch_top + patch_height

    patch_image = (
        image[:, patch_top:patch_bottom, patch_left:patch_right].clone().detach()
    )
    patch_image = image_transform(patch_image)  # [C, H, W]

    augmented_image = image.clone().detach()
    if mode == "normal":
        new_right = new_left + patch_width
        new_bottom = new_top + patch_height
        mask = torch.zeros_like(image[0:1, :, :])  # [1, H, W]
        mask[:, new_top:new_bottom, new_left:new_right] = 1.0
        augmented_image[:, new_top:new_bottom, new_left:new_right] = patch_image
    elif mode == "scar":
        alpha = torch.ones(1, patch_image.size(1), patch_image.size(2), device=device)
        patch_image = torch.cat([patch_image, alpha], dim=0)  # [C+1, H, W]

        if rotation_transform is not None:
            patch_image = rotation_transform(patch_image)  # [C+1, rotH, rotW]

        patch_height = patch_image.size(1)
        patch_width = patch_image.size(2)
        pad = (
            new_left,
            original_width - new_left - patch_width,
            new_top,
            original_height - new_top - patch_height,
        )

        mask = F.pad(patch_image[-1:], pad)
        patch_image = F.pad(patch_image[:-1], pad)
        augmented_image = image * (1.0 - mask) + patch_image * mask
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")

    return augmented_image, mask


class CutPasteNormal(v2.Transform):

    def __init__(
        self,
        probability: float = 0.5,
        area_ratio: tuple[float, float] = (0.02, 0.15),
        aspect_ratio: list[tuple[float, float]] = [(0.3, 1), (1, 3.3)],
        max_intensity: float = 0.1,
    ) -> None:
        super().__init__()
        self.probability = probability
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

        self.color_jitter = v2.ColorJitter(
            brightness=max_intensity,
            contrast=max_intensity,
            saturation=max_intensity,
            hue=max_intensity,
        )

    def _transform_image(
        self,
        img: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if torch.rand(1, device=device) > self.probability:
            return img, torch.zeros((1, h, w), device=device)

        # calculate patch area randomly within the specified range
        image_area = h * w
        patch_area = (
            torch.empty(1, device=device).uniform_(*self.area_ratio).item() * image_area
        )

        # choose aspect ratio range randomly
        ratio_index = torch.randint(len(self.aspect_ratio), (1,), device=device).item()
        # calculate patch aspect ratio randomly within the chosen range
        patch_aspect = (
            torch.empty(1, device=device)
            .uniform_(*self.aspect_ratio[ratio_index])
            .item()
        )

        # calculate patch width and height based on area and aspect ratio
        patch_width = int(round((patch_area * patch_aspect) ** 0.5))
        patch_height = int(round((patch_area / patch_aspect) ** 0.5))

        # ensure width and height are within bounds
        patch_width = max(1, min(patch_width, w // 2))
        patch_height = max(1, min(patch_height, h // 2))

        return crop_and_patch(
            img, patch_width, patch_height, self.color_jitter, "normal", device
        )

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_batch = len(img.shape) == 4
        if is_batch:
            device = img.device
            batch, _, height, width = img.shape
            batch_augmented = []
            batch_masks = []

            for i in range(batch):
                augmented, mask = self._transform_image(img[i], height, width, device)
                batch_augmented.append(augmented)
                batch_masks.append(mask)

            return torch.stack(batch_augmented), torch.stack(batch_masks)

        return self._transform_image(img, img.shape[1], img.shape[2], img.device)


class CutPasteScar(v2.Transform):
    def __init__(
        self,
        probability: float = 0.5,
        width_range: tuple[float, float] = (2, 16),
        length_range: tuple[float, float] = (10, 25),
        max_intensity: float = 0.1,
        roatation_range: float | tuple[float, float] = (-45, 45),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.width_range = width_range
        self.length_range = length_range
        self.max_intensity = max_intensity
        self.rotation_range = roatation_range

        self.color_jitter = v2.ColorJitter(
            brightness=max_intensity,
            contrast=max_intensity,
            saturation=max_intensity,
            hue=max_intensity,
        )
        self.rotation = v2.RandomRotation(
            degrees=self.rotation_range,
            expand=True,
            fill=0,
        )

    def _transform_image(
        self,
        image: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if torch.rand(1, device=device) > self.probability:
            return image, torch.zeros((1, h, w), device=device)

        patch_width = int(
            torch.empty(1, device=device).uniform_(*self.width_range).item()
        )
        patch_height = int(
            torch.empty(1, device=device).uniform_(*self.length_range).item()
        )

        patch_width = max(1, min(patch_width, w // 2))
        patch_height = max(1, min(patch_height, h // 2))

        return crop_and_patch(
            image,
            patch_width,
            patch_height,
            self.color_jitter,
            "scar",
            device,
            rotation_transform=self.rotation,
        )

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_batch = len(img.shape) == 4
        if is_batch:
            device = img.device
            batch, _, height, width = img.shape
            batch_augmented = []
            batch_masks = []

            for i in range(batch):
                augmented, mask = self._transform_image(img[i], height, width, device)
                batch_augmented.append(augmented)
                batch_masks.append(mask)

            return torch.stack(batch_augmented), torch.stack(batch_masks)

        return self._transform_image(img, img.shape[1], img.shape[2], img.device)
