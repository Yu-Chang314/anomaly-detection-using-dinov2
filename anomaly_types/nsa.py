import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2

# patch_ex / _patch_ex are taken from the official NSA implementation (MIT License):
# https://github.com/hmsch/natural-synthetic-anomalies/blob/master/self_sup_data/self_sup_tasks.py
# Copyright (c) 2021 Hannah Schlueter
# Schlueter et al., "Natural Synthetic Anomalies for Self-Supervised Anomaly
# Detection and Localization", ECCV 2022.


def patch_ex(ima_dest, ima_src=None, same=False, num_patches=1,
             mode=cv2.NORMAL_CLONE, width_bounds_pct=((0.05, 0.2), (0.05, 0.2)), min_object_pct=0.25,
             min_overlap_pct=0.25, shift=True, label_mode='binary', skip_background=None, tol=1, resize=True,
             gamma_params=None, intensity_logistic_params=(1 / 6, 20),
             resize_bounds=(0.7, 1.3), num_ellipses=None, verbose=True, cutpaste_patch_generation=False):
    """
    Create a synthetic training example from the given images by pasting/blending random patches.
    Args:
        ima_dest (uint8 numpy array): image with shape (W,H,3) or (W,H,1) where patch should be changed
        ima_src (uint8 numpy array): optional, otherwise use ima_dest as source
        same (bool): use ima_dest as source even if ima_src given
        mode: 'uniform', 'swap', 'mix', cv2.NORMAL_CLONE, or cv2.MIXED_CLONE what blending method to use
             ('mix' is flip a coin between normal and mixed clone)
        num_patches (int): how many patches to add. the method will always attempt to add the first patch,
                    for each subsequent patch it flips a coin
        width_bounds_pct ((float, float), (float, float)): min half-width of patch ((min_dim1, max_dim1), (min_dim2, max_dim2))
        shift (bool): if false, patches in src and dest image have same coords. otherwise random shift
        resize (bool): if true, patch is resampled at random size (within bounds and keeping aspect ratio the same) before blending
        skip_background (int, int) or [(int, int),]: optional, assume background color is first and only interpolate patches
                    in areas where dest or src patch has pixelwise MAD < second from background.
        tol (int): mean abs intensity change required to get positive label
        gamma_params (float, float, float): optional, (shape, scale, left offset) of gamma dist to sample half-width of patch from,
                    otherwise use uniform dist between 0.05 and 0.95
        intensity_logistic_params (float, float): k, x0 of logitistc map for intensity based label
        num_ellipses (int): optional, if set, the rectangular patch mask is filled with random ellipses
        label_mode: 'binary',
                    'continuous' -- use interpolation factor as label (only when mode is 'uniform'),
                    'intensity' -- use median filtered mean absolute pixelwise intensity difference as label,
                    'logistic-intensity' -- use logistic median filtered of mean absolute pixelwise intensity difference as label,
        cutpaste_patch_generation (bool): optional, if set, width_bounds_pct, resize, skip_background, min_overlap_pct, min_object_pct,
                    num_patches and gamma_params are ignored. A single patch is sampled as in the CutPaste paper:
                        1. sampling the area ratio between the patch and the full image from (0.02, 0.15)
                        2. determine the aspect ratio by sampling from (0.3, 1) union (1, 3.3)
                        3. sample location such that patch is contained entirely within the image
    """
    if mode == 'mix':
        mode = (cv2.NORMAL_CLONE, cv2.MIXED_CLONE)[np.random.randint(2)]

    if cutpaste_patch_generation:
        width_bounds_pct = None
        resize = False
        skip_background = None
        min_overlap_pct = None
        min_object_pct = None
        gamma_params = None
        num_patches = 1

    ima_src = ima_dest.copy() if same or (ima_src is None) else ima_src

    if skip_background is not None and not cutpaste_patch_generation:
        if isinstance(skip_background, tuple):
            skip_background = [skip_background]
        src_object_mask = np.ones_like(ima_src[..., 0:1])
        dest_object_mask = np.ones_like(ima_dest[..., 0:1])
        for background, threshold in skip_background:
            src_object_mask &= np.uint8(np.abs(ima_src.mean(axis=-1, keepdims=True) - background) > threshold)
            dest_object_mask &= np.uint8(np.abs(ima_dest.mean(axis=-1, keepdims=True) - background) > threshold)
        src_object_mask[..., 0] = cv2.medianBlur(src_object_mask[..., 0], 7)  # remove grain from threshold choice
        dest_object_mask[..., 0] = cv2.medianBlur(dest_object_mask[..., 0], 7)  # remove grain from threshold choice
    else:
        src_object_mask = None
        dest_object_mask = None

    # add patches
    mask = np.zeros_like(ima_dest[..., 0:1])  # single channel
    patchex = ima_dest.copy()
    coor_min_dim1, coor_max_dim1, coor_min_dim2, coor_max_dim2 = mask.shape[0] - 1, 0, mask.shape[1] - 1, 0
    if label_mode == 'continuous':
        factor = np.random.uniform(0.05, 0.95)
    else:
        factor = 1
    for i in range(num_patches):
        if i == 0 or np.random.randint(2) > 0:  # at least one patch
            patchex, ((_coor_min_dim1, _coor_max_dim1), (_coor_min_dim2, _coor_max_dim2)), patch_mask = _patch_ex(
                patchex, ima_src, dest_object_mask, src_object_mask, mode, label_mode, shift, resize, width_bounds_pct,
                gamma_params, min_object_pct, min_overlap_pct, factor, resize_bounds, num_ellipses, verbose, cutpaste_patch_generation)
            if patch_mask is not None:
                mask[_coor_min_dim1:_coor_max_dim1, _coor_min_dim2:_coor_max_dim2] = patch_mask
                coor_min_dim1 = min(coor_min_dim1, _coor_min_dim1)
                coor_max_dim1 = max(coor_max_dim1, _coor_max_dim1)
                coor_min_dim2 = min(coor_min_dim2, _coor_min_dim2)
                coor_max_dim2 = max(coor_max_dim2, _coor_max_dim2)

    # create label
    label_mask = np.uint8(np.mean(np.abs(1.0 * mask * ima_dest - 1.0 * mask * patchex), axis=-1, keepdims=True) > tol)
    label_mask[..., 0] = cv2.medianBlur(label_mask[..., 0], 5)  # remove grain from threshold choice

    if label_mode == 'continuous':
        label = label_mask * factor
    elif label_mode in ['logistic-intensity', 'intensity']:
        from skimage.morphology import disk
        from skimage.filters import median
        k, x0 = intensity_logistic_params
        label = np.mean(np.abs(label_mask * ima_dest * 1.0 - label_mask * patchex * 1.0), axis=-1, keepdims=True)
        label[..., 0] = median(label[..., 0], disk(5))
        if label_mode == 'logistic-intensity':
            label = label_mask / (1 + np.exp(-k * (label - x0)))
    elif label_mode == 'binary':
        label = label_mask
    else:
        raise ValueError("label_mode not supported" + str(label_mode))

    return patchex, label


def _patch_ex(ima_dest, ima_src, dest_object_mask, src_object_mask, mode, label_mode, shift, resize, width_bounds_pct,
              gamma_params, min_object_pct, min_overlap_pct, factor, resize_bounds, num_ellipses, verbose, cutpaste_patch_generation):
    if cutpaste_patch_generation:
        skip_background = False
        dims = np.array(ima_dest.shape)
        if dims[0] != dims[1]:
            raise ValueError("CutPaste patch generation only works for square images")
        # 1. sampling the area ratio between the patch and the full image from (0.02, 0.15)
        # (divide by 4 as patch-widths below are actually half-widths)
        area_ratio = np.random.uniform(0.02, 0.15) / 4.0
        #  2. determine the aspect ratio by sampling from (0.3, 1) union (1, 3.3)
        if np.random.randint(2) > 0:
            aspect_ratio = np.random.uniform(0.3, 1)
        else:
            aspect_ratio = np.random.uniform(1, 3.3)

        patch_width_dim1 = int(np.rint(np.clip(np.sqrt(area_ratio * aspect_ratio * dims[0] ** 2), 0, dims[0])))
        patch_width_dim2 = int(np.rint(np.clip(area_ratio * dims[0] ** 2 / patch_width_dim1, 0, dims[1])))
        #  3. sample location such that patch is contained entirely within the image
        center_dim1 = np.random.randint(patch_width_dim1, dims[0] - patch_width_dim1)
        center_dim2 = np.random.randint(patch_width_dim2, dims[1] - patch_width_dim2)

        coor_min_dim1 = np.clip(center_dim1 - patch_width_dim1, 0, dims[0])
        coor_min_dim2 = np.clip(center_dim2 - patch_width_dim2, 0, dims[1])
        coor_max_dim1 = np.clip(center_dim1 + patch_width_dim1, 0, dims[0])
        coor_max_dim2 = np.clip(center_dim2 + patch_width_dim2, 0, dims[1])

        patch_mask = np.ones((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2, 1), dtype=np.uint8)
    else:
        skip_background = (src_object_mask is not None) and (dest_object_mask is not None)
        dims = np.array(ima_dest.shape)
        min_width_dim1 = (width_bounds_pct[0][0] * dims[0]).round().astype(int)
        max_width_dim1 = (width_bounds_pct[0][1] * dims[0]).round().astype(int)
        min_width_dim2 = (width_bounds_pct[1][0] * dims[1]).round().astype(int)
        max_width_dim2 = (width_bounds_pct[1][1] * dims[1]).round().astype(int)

        if gamma_params is not None:
            shape, scale, lower_bound = gamma_params
            patch_width_dim1 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[0], min_width_dim1, max_width_dim1))
            patch_width_dim2 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[1], min_width_dim2, max_width_dim2))
        else:
            patch_width_dim1 = np.random.randint(min_width_dim1, max_width_dim1)
            patch_width_dim2 = np.random.randint(min_width_dim2, max_width_dim2)

        found_patch = False
        attempts = 0
        while not found_patch:
            center_dim1 = np.random.randint(min_width_dim1, dims[0] - min_width_dim1)
            center_dim2 = np.random.randint(min_width_dim2, dims[1] - min_width_dim2)

            coor_min_dim1 = np.clip(center_dim1 - patch_width_dim1, 0, dims[0])
            coor_min_dim2 = np.clip(center_dim2 - patch_width_dim2, 0, dims[1])
            coor_max_dim1 = np.clip(center_dim1 + patch_width_dim1, 0, dims[0])
            coor_max_dim2 = np.clip(center_dim2 + patch_width_dim2, 0, dims[1])

            if num_ellipses is not None:
                ellipse_min_dim1 = min_width_dim1
                ellipse_min_dim2 = min_width_dim2
                ellipse_max_dim1 = max(min_width_dim1 + 1, patch_width_dim1 // 2)
                ellipse_max_dim2 = max(min_width_dim2 + 1, patch_width_dim2 // 2)
                patch_mask = np.zeros((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2), dtype=np.uint8)
                x = np.arange(patch_mask.shape[0]).reshape(-1, 1)
                y = np.arange(patch_mask.shape[1]).reshape(1, -1)
                for _ in range(num_ellipses):
                    theta = np.random.uniform(0, np.pi)
                    x0 = np.random.randint(0, patch_mask.shape[0])
                    y0 = np.random.randint(0, patch_mask.shape[1])
                    a = np.random.randint(ellipse_min_dim1, ellipse_max_dim1)
                    b = np.random.randint(ellipse_min_dim2, ellipse_max_dim2)
                    ellipse = (((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) / a) ** 2 + (
                        ((x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) / b) ** 2 <= 1  # True for points inside the ellipse
                    patch_mask |= ellipse
                patch_mask = patch_mask[..., None]
            else:
                patch_mask = np.ones((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2, 1), dtype=np.uint8)

            if skip_background:
                background_area = np.sum(patch_mask & src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2])
                if num_ellipses is not None:
                    patch_area = np.sum(patch_mask)
                else:
                    patch_area = patch_mask.shape[0] * patch_mask.shape[1]
                found_patch = (background_area / patch_area > min_object_pct)
            else:
                found_patch = True
            attempts += 1
            if attempts == 200:
                if verbose:
                    print('No suitable patch found.')
                return ima_dest.copy(), ((0, 0), (0, 0)), None

    src = ima_src[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
    height, width, _ = src.shape
    if resize:
        lb, ub = resize_bounds
        scale = np.clip(np.random.normal(1, 0.5), lb, ub)
        new_height = np.clip(scale * height, min_width_dim1, max_width_dim1)
        new_width = np.clip(int(new_height / height * width), min_width_dim2, max_width_dim2)
        new_height = np.clip(int(new_width / width * height), min_width_dim1, max_width_dim1)  # in case there was clipping
        if src.shape[2] == 1:  # grayscale
            src = cv2.resize(src[..., 0], (new_width, new_height))
            src = src[..., None]
        else:
            src = cv2.resize(src, (new_width, new_height))
        height, width, _ = src.shape
        patch_mask = cv2.resize(patch_mask[..., 0], (width, height))
        patch_mask = patch_mask[..., None]

    if skip_background:
        src_object_mask = cv2.resize(src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2, 0], (width, height))
        src_object_mask = src_object_mask[..., None]

    # sample destination location and size
    if shift:
        found_center = False
        attempts = 0
        while not found_center:
            center_dim1 = np.random.randint(height // 2 + 1, ima_dest.shape[0] - height // 2 - 1)
            center_dim2 = np.random.randint(width // 2 + 1, ima_dest.shape[1] - width // 2 - 1)
            coor_min_dim1, coor_max_dim1 = center_dim1 - height // 2, center_dim1 + (height + 1) // 2
            coor_min_dim2, coor_max_dim2 = center_dim2 - width // 2, center_dim2 + (width + 1) // 2

            if skip_background:
                src_and_dest = dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] & src_object_mask & patch_mask
                src_or_dest = (dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] | src_object_mask) & patch_mask
                found_center = (np.sum(src_object_mask) / (patch_mask.shape[0] * patch_mask.shape[1]) > min_object_pct and    # contains object
                                np.sum(src_and_dest) / np.sum(src_object_mask) > min_overlap_pct)                    # object overlaps src object
            else:
                found_center = True
            attempts += 1
            if attempts == 200:
                if verbose:
                    print('No suitable center found. Dims were:', width, height)
                return ima_dest.copy(), ((0, 0), (0, 0)), None

    # blend
    if skip_background:
        patch_mask &= src_object_mask | dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]

    if mode == 'swap':
        patchex = ima_dest.copy()
        before = patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] -= patch_mask * before
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] += patch_mask * src
    elif mode == 'uniform':
        patchex = 1.0 * ima_dest
        before = patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] -= factor * patch_mask * before
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] += factor * patch_mask * src
        patchex = np.uint8(np.floor(patchex))
    elif mode in [cv2.NORMAL_CLONE, cv2.MIXED_CLONE]:  # poisson interpolation
        int_factor = np.uint8(np.ceil(factor * 255))
        # add background to patchmask to avoid artefacts
        if skip_background:
            patch_mask_scaled = int_factor * (patch_mask | ((1 - src_object_mask) & (1 - dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2])))
        else:
            patch_mask_scaled = int_factor * patch_mask
        patch_mask_scaled[0], patch_mask_scaled[-1], patch_mask_scaled[:, 0], patch_mask_scaled[:, -1] = 0, 0, 0, 0  # zero border to avoid artefacts
        center = (coor_max_dim2 - (coor_max_dim2 - coor_min_dim2) // 2, coor_min_dim1 + (coor_max_dim1 - coor_min_dim1) // 2)  # height dim first
        if np.sum(patch_mask_scaled > 0) < 50:  # cv2 seamlessClone will fail if positive mask area is too small
            return ima_dest.copy(), ((0, 0), (0, 0)), None
        try:
            if ima_dest.shape[2] == 1:  # grayscale
                # pad to 3 channels as that's what OpenCV expects
                src_3 = np.concatenate((src, np.zeros_like(src), np.zeros_like(src)), axis=2)
                ima_dest_3 = np.concatenate((ima_dest, np.zeros_like(ima_dest), np.zeros_like(ima_dest)), axis=2)
                patchex = cv2.seamlessClone(src_3, ima_dest_3, patch_mask_scaled, center, mode)
                patchex = patchex[..., 0:1]  # extract first channel
            else:  # RGB
                patchex = cv2.seamlessClone(src, ima_dest, patch_mask_scaled, center, mode)
        except cv2.error as e:
            print('WARNING, tried bad interpolation mask and got:', e)
            return ima_dest.copy(), ((0, 0), (0, 0)), None
    else:
        raise ValueError("mode not supported" + str(mode))

    return patchex, ((coor_min_dim1, coor_max_dim1), (coor_min_dim2, coor_max_dim2)), patch_mask


# Per-class hyperparameters from the official NSA repo (train_mvtec.py).
# Widths are half-widths relative to the image size.
WIDTH_BOUNDS_PCT = {'bottle': ((0.03, 0.4), (0.03, 0.4)), 'cable': ((0.05, 0.4), (0.05, 0.4)), 'capsule': ((0.03, 0.15), (0.03, 0.4)),
                    'hazelnut': ((0.03, 0.35), (0.03, 0.35)), 'metal_nut': ((0.03, 0.4), (0.03, 0.4)), 'pill': ((0.03, 0.2), (0.03, 0.4)),
                    'screw': ((0.03, 0.12), (0.03, 0.12)), 'toothbrush': ((0.03, 0.4), (0.03, 0.2)), 'transistor': ((0.03, 0.4), (0.03, 0.4)),
                    'zipper': ((0.03, 0.4), (0.03, 0.2)),
                    'carpet': ((0.03, 0.4), (0.03, 0.4)), 'grid': ((0.03, 0.4), (0.03, 0.4)),
                    'leather': ((0.03, 0.4), (0.03, 0.4)), 'tile': ((0.03, 0.4), (0.03, 0.4)), 'wood': ((0.03, 0.4), (0.03, 0.4))}

MIN_OVERLAP_PCT = {'bottle': 0.25, 'capsule': 0.25, 'hazelnut': 0.25, 'metal_nut': 0.25,
                   'pill': 0.25, 'screw': 0.25, 'toothbrush': 0.25, 'zipper': 0.25}

MIN_OBJECT_PCT = {'bottle': 0.7, 'capsule': 0.7, 'hazelnut': 0.7, 'metal_nut': 0.5,
                  'pill': 0.7, 'screw': 0.5, 'toothbrush': 0.25, 'zipper': 0.7}

NUM_PATCHES = {'bottle': 3, 'cable': 3, 'capsule': 3, 'hazelnut': 3, 'metal_nut': 3,
               'pill': 3, 'screw': 4, 'toothbrush': 3, 'transistor': 3, 'zipper': 4,
               'carpet': 4, 'grid': 4, 'leather': 4, 'tile': 4, 'wood': 4}

# (brightness, threshold) pairs
BACKGROUND = {'bottle': (200, 60), 'screw': (200, 60), 'capsule': (200, 60), 'zipper': (200, 60),
              'hazelnut': (20, 20), 'pill': (20, 20), 'toothbrush': (20, 20), 'metal_nut': (20, 20)}

TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


def mvtec_patch_ex_args(class_name: str) -> dict:
    """Build patch_ex keyword arguments for an MVTec class using the official NSA settings."""
    args = {
        'width_bounds_pct': WIDTH_BOUNDS_PCT[class_name],
        'num_patches': NUM_PATCHES[class_name],
        'gamma_params': (2, 0.05, 0.03),
        'resize': True,
        'shift': True,
        'same': False,
        'label_mode': 'binary',
        'skip_background': BACKGROUND.get(class_name),
        'min_object_pct': MIN_OBJECT_PCT.get(class_name, 0.25),
        'min_overlap_pct': MIN_OVERLAP_PCT.get(class_name, 0.25),
        'verbose': False,
    }
    if class_name in TEXTURES:
        # per NSA paper appendix: mixed gradients for textures, source gradients for objects
        args['mode'] = cv2.MIXED_CLONE
        args['resize_bounds'] = (0.5, 2.0)
    else:
        args['mode'] = cv2.NORMAL_CLONE
    return args


class NSAAnomalyGenerator(v2.Transform):
    """Wraps NSA patch_ex as a v2.Transform matching the CutPaste/Perlin generator interface.

    Blends a randomly sampled patch from another normal training image into the
    input image via Poisson image editing (seamlessClone).
    """

    def __init__(
        self,
        class_name: str,
        source_dir: Path | str,
        probability: float = 0.5,
        resize: int = 256,
        cropsize: int = 224,
    ) -> None:
        super().__init__()
        self.probability = probability
        self.patch_ex_args = mvtec_patch_ex_args(class_name)

        self.source_paths = sorted(Path(source_dir).glob("*.png"))
        if not self.source_paths:
            raise FileNotFoundError(f"No source images found in {source_dir}")

        self.transform_src = v2.Compose(
            [
                v2.Resize(size=resize, interpolation=v2.InterpolationMode.LANCZOS),
                v2.CenterCrop(cropsize),
            ]
        )

    def _load_source(self) -> np.ndarray:
        idx = int(torch.randint(len(self.source_paths), (1,)).item())
        src = Image.open(self.source_paths[idx]).convert("RGB")
        src = self.transform_src(src)
        return np.asarray(src)

    def _transform_image(
        self, img: torch.Tensor, h: int, w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1, device=device) > self.probability:
            return img, torch.zeros((1, h, w), device=device)

        dest = (img.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        src = self._load_source()
        patchex, label = patch_ex(dest, src, **self.patch_ex_args)

        augmented = (
            torch.from_numpy(patchex.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(device)
        )
        mask = (
            torch.from_numpy(label.astype(np.float32))
            .permute(2, 0, 1)
            .to(device)
        )
        return augmented, mask

    def forward(
        self, img: torch.Tensor
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
