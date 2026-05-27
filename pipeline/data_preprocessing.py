from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps


EYEBROW_LABELS = (2, 3)  # CelebAMask-HQ: left_brow, right_brow
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LEFT_EYEBROW_LANDMARKS = (105, 66, 107, 52, 53, 65, 55, 70, 63, 46)
RIGHT_EYEBROW_LANDMARKS = (334, 296, 336, 282, 283, 295, 285, 300, 293, 276)
LEFT_EYE_LANDMARKS = (33, 133, 159, 145)
RIGHT_EYE_LANDMARKS = (362, 263, 386, 374)
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
WEIGHT_URLS = [
    "https://huggingface.co/AI2lab/face-parsing.PyTorch/resolve/main/79999_iter.pth?download=true",
    "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth?download=true",
]


def conv3x3(in_chan: int, out_chan: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_chan,
        out_chan,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chan, out_chan, 1)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.downsample = None
        if stride != 1 or in_chan != out_chan:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


def create_resnet18() -> torchvision.models.ResNet:
    try:
        return torchvision.models.resnet18(weights=None)
    except TypeError:
        return torchvision.models.resnet18(pretrained=False)


class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = create_resnet18()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat8 = self.layer2(self.layer1(x))
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat32.shape[2:], mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = feat * atten
        return feat_atten + feat


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    def __init__(self, n_classes: int = 19) -> None:
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        out = self.conv_out(feat_fuse)
        out16 = self.conv_out16(feat_cp8)
        out32 = self.conv_out32(feat_cp16)

        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        out16 = F.interpolate(
            out16, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        out32 = F.interpolate(
            out32, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        return out, out16, out32


@dataclass
class MaskStats:
    image_name: str
    width: int
    height: int
    tight_pixels: int
    bbox_xyxy: list[int] | None
    face_bbox_xyxy: list[int] | None
    padded_pixels: dict[str, int]
    padded_bboxes_xyxy: dict[str, list[int] | None]


def safe_output_stem(path: Path) -> str:
    raw = f"{path.stem}_{path.suffix.lower().lstrip('.')}"
    safe_chars = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    stem = "".join(safe_chars).strip("_")
    return stem or "image"


def build_output_stem_map(input_dir: Path) -> dict[Path, str]:
    paths = list(iter_images(input_dir))
    stem_counts: dict[str, int] = {}
    for path in paths:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    mapping: dict[Path, str] = {}
    for path in paths:
        if stem_counts[path.stem] == 1:
            mapping[path] = path.stem
        else:
            mapping[path] = f"{path.stem}_{path.suffix.lower().lstrip('.')}"
    return mapping


def ensure_mediapipe_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path
    response = requests.get(MEDIAPIPE_MODEL_URL, stream=True, timeout=60)
    response.raise_for_status()
    with model_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return model_path


def create_face_landmarker(model_path: Path) -> vision.FaceLandmarker:
    ensure_mediapipe_model(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(image)


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def ensure_weights(weight_path: Path, force_download: bool = False) -> Path:
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if weight_path.exists() and not force_download:
        return weight_path

    errors: list[str] = []
    for url in WEIGHT_URLS:
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with weight_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            return weight_path
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "BiSeNet 가중치 다운로드에 실패했습니다. "
        f"`--weights`로 직접 지정하거나 아래 URL 중 하나에서 받아주세요:\n{joined}"
    )
    return weight_path


def load_model(weight_path: Path, device: torch.device) -> BiSeNet:
    model = BiSeNet(n_classes=19)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def get_face_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )


def get_profile_face_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
    )


def get_eye_detectors() -> list[cv2.CascadeClassifier]:
    names = [
        "haarcascade_eye_tree_eyeglasses.xml",
        "haarcascade_eye.xml",
    ]
    return [
        cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        for name in names
    ]


def connected_component_cleanup(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or mask.max() == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept[labels == label] = 255
    return kept


def clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def detect_primary_face(
    image_rgb: np.ndarray,
    detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier | None = None,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    candidates: list[tuple[int, int, int, int]] = []
    for scale in (1.0, 1.5):
        scan = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale)
        min_size = (max(48, int(80 * scale)), max(48, int(80 * scale)))
        faces = detector.detectMultiScale(
            scan,
            scaleFactor=1.08,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size,
        )
        for x, y, w, h in faces:
            if scale != 1.0:
                x, y, w, h = (
                    int(round(x / scale)),
                    int(round(y / scale)),
                    int(round(w / scale)),
                    int(round(h / scale)),
                )
            candidates.append((x, y, w, h))

    if profile_detector is not None:
        for flipped in (False, True):
            scan = gray if not flipped else cv2.flip(gray, 1)
            faces = profile_detector.detectMultiScale(
                scan,
                scaleFactor=1.08,
                minNeighbors=4,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(60, 60),
            )
            for x, y, w, h in faces:
                if flipped:
                    x = gray.shape[1] - x - w
                candidates.append((int(x), int(y), int(w), int(h)))

    if not candidates:
        return None
    x, y, w, h = max(candidates, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(x + w), int(y + h)


def expand_face_box(
    face_box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = face_box
    w = x2 - x1
    h = y2 - y1
    face_ratio = (w * h) / max(width * height, 1)
    if face_ratio < 0.08:
        x_pad, top_pad, bottom_pad = 0.42, 0.46, 0.32
    elif face_ratio < 0.16:
        x_pad, top_pad, bottom_pad = 0.34, 0.40, 0.27
    else:
        x_pad, top_pad, bottom_pad = 0.28, 0.34, 0.22
    expanded = (
        int(round(x1 - w * x_pad)),
        int(round(y1 - h * top_pad)),
        int(round(x2 + w * x_pad)),
        int(round(y2 + h * bottom_pad)),
    )
    return clip_box(expanded, width, height)


def crop_array(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def paste_crop_mask(
    crop_mask: np.ndarray, full_shape: tuple[int, int], box: tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = box
    full = np.zeros(full_shape, dtype=np.uint8)
    full[y1:y2, x1:x2] = crop_mask
    return full


def rotate_image_by_k(image: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return image
    return np.rot90(image, k=k)


def unrotate_image_by_k(image: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return image
    return np.rot90(image, k=(4 - k) % 4)


def rotate_image_by_angle(
    image: np.ndarray,
    angle_deg: float,
    interpolation: int,
    border_mode: int,
    border_value: int | tuple[int, int, int] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )
    return rotated, matrix


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def dilate_mask(mask: np.ndarray, padding_px: int) -> np.ndarray:
    if padding_px <= 0:
        return mask.copy()
    kernel_size = padding_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def build_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image_rgb.copy()
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask > 0] = (
        overlay[mask > 0].astype(np.float32) * 0.4 + color_arr.astype(np.float32) * 0.6
    ).astype(np.uint8)
    return overlay


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask).save(path)


def extract_foreground_on_white(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full_like(image_rgb, 255)
    result[mask > 0] = image_rgb[mask > 0]
    return result


def infer_parsing_map(
    model: BiSeNet,
    image: Image.Image,
    device: torch.device,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = image_to_tensor(image, input_size).to(device)
    with torch.no_grad():
        logits, _, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Horizontal-flip ensemble helps when one eyebrow is weaker than the other
        # due to pose, lighting, or model asymmetry.
        flipped_tensor = torch.flip(tensor, dims=[3])
        flipped_logits, _, _ = model(flipped_tensor)
        flipped_probs = torch.softmax(flipped_logits, dim=1).squeeze(0).cpu().numpy()
        flipped_probs = np.flip(flipped_probs, axis=2)
        probs = (probs + flipped_probs) / 2.0

        parsing = probs.argmax(axis=0).astype(np.uint8)
        brow_prob = probs[2] + probs[3]
    orig_w, orig_h = image.size
    parsing = cv2.resize(parsing, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    brow_prob = cv2.resize(brow_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return parsing, brow_prob


def component_score(
    stats_row: np.ndarray,
    centroid: tuple[float, float],
    half: str,
    width: int,
    height: int,
) -> float:
    x = stats_row[cv2.CC_STAT_LEFT]
    y = stats_row[cv2.CC_STAT_TOP]
    w = stats_row[cv2.CC_STAT_WIDTH]
    h = stats_row[cv2.CC_STAT_HEIGHT]
    area = stats_row[cv2.CC_STAT_AREA]
    cx, cy = centroid
    if h <= 0 or w <= 0:
        return -1e9
    aspect = w / max(h, 1)
    aspect_bonus = 1.0 if 1.3 <= aspect <= 8.0 else -0.7
    vertical_target = height * 0.31
    vertical_score = 1.0 - min(abs(cy - vertical_target) / max(height * 0.22, 1.0), 1.4)
    horizontal_target = width * (0.32 if half == "left" else 0.68)
    horizontal_score = 1.0 - min(
        abs(cx - horizontal_target) / max(width * 0.22, 1.0),
        1.4,
    )
    size_score = min(area / max(width * height * 0.008, 1.0), 1.2)
    eyebrow_band_bonus = 0.4 if y < height * 0.45 and (y + h) > height * 0.15 else -0.8
    return aspect_bonus + vertical_score + horizontal_score + size_score + eyebrow_band_bonus


def detect_eye_boxes(
    face_rgb: np.ndarray,
    eye_detectors: list[cv2.CascadeClassifier],
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    upper = gray[: max(1, int(h * 0.62)), :]
    upsample = 1.8 if min(h, w) < 320 else 1.0
    scan = upper if upsample == 1.0 else cv2.resize(upper, None, fx=upsample, fy=upsample)
    candidates: list[tuple[int, int, int, int]] = []
    for detector in eye_detectors:
        eyes = detector.detectMultiScale(
            scan,
            scaleFactor=1.05,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(max(12, int(scan.shape[1] * 0.06)), max(8, int(scan.shape[0] * 0.04))),
        )
        for x, y, ew, eh in eyes:
            if upsample != 1.0:
                x, y, ew, eh = (
                    int(round(x / upsample)),
                    int(round(y / upsample)),
                    int(round(ew / upsample)),
                    int(round(eh / upsample)),
                )
            if y > h * 0.48:
                continue
            if ew <= eh:
                continue
            candidates.append((int(x), int(y), int(ew), int(eh)))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(candidates, key=lambda item: item[2] * item[3], reverse=True):
        bx, by, bw, bh = box
        cx = bx + bw / 2
        cy = by + bh / 2
        if any(abs(cx - (dx + dw / 2)) < bw * 0.4 and abs(cy - (dy + dh / 2)) < bh * 0.4 for dx, dy, dw, dh in deduped):
            continue
        deduped.append(box)
    return deduped[:8]


def pick_eye_pair(
    eye_boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if len(eye_boxes) < 2:
        return None
    best_pair = None
    best_score = -1e9
    for i in range(len(eye_boxes)):
        for j in range(i + 1, len(eye_boxes)):
            a = eye_boxes[i]
            b = eye_boxes[j]
            if a[0] > b[0]:
                a, b = b, a
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            acx, acy = ax + aw / 2, ay + ah / 2
            bcx, bcy = bx + bw / 2, by + bh / 2
            if bcx - acx < width * 0.12:
                continue
            if abs(acy - bcy) > height * 0.12:
                continue
            size_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh, 1)
            center_span = bcx - acx
            span_score = 1.0 - min(abs(center_span - width * 0.34) / max(width * 0.25, 1), 1.0)
            vertical_score = 1.0 - min(abs(((acy + bcy) / 2) - height * 0.40) / max(height * 0.2, 1), 1.0)
            score = size_ratio + span_score + vertical_score
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    return best_pair


def build_eye_guided_roi_mask(
    height: int,
    width: int,
    eye_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None,
) -> np.ndarray:
    if eye_pair is None:
        return build_face_roi_mask(height, width)
    roi = np.zeros((height, width), dtype=np.uint8)
    for eye in eye_pair:
        x, y, ew, eh = eye
        brow_x1 = int(round(x - ew * 0.18))
        brow_x2 = int(round(x + ew * 1.18))
        brow_y1 = int(round(y - eh * 1.55))
        brow_y2 = int(round(y - eh * 0.05))
        brow_x1, brow_y1, brow_x2, brow_y2 = clip_box((brow_x1, brow_y1, brow_x2, brow_y2), width, height)
        roi[brow_y1:brow_y2, brow_x1:brow_x2] = 255
    return roi


def build_eye_guided_half_rois(
    height: int,
    width: int,
    eye_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None,
) -> dict[str, np.ndarray]:
    if eye_pair is None:
        base = build_face_roi_mask(height, width)
        mid_x = width // 2
        left = np.zeros_like(base)
        right = np.zeros_like(base)
        left[:, :mid_x] = base[:, :mid_x]
        right[:, mid_x:] = base[:, mid_x:]
        return {"left": left, "right": right}

    rois: dict[str, np.ndarray] = {}
    for side, eye in zip(("left", "right"), eye_pair):
        roi = np.zeros((height, width), dtype=np.uint8)
        x, y, ew, eh = eye
        brow_x1 = int(round(x - ew * 0.28))
        brow_x2 = int(round(x + ew * 1.28))
        brow_y1 = int(round(y - eh * 1.80))
        brow_y2 = int(round(y + eh * 0.18))
        brow_x1, brow_y1, brow_x2, brow_y2 = clip_box(
            (brow_x1, brow_y1, brow_x2, brow_y2),
            width,
            height,
        )
        roi[brow_y1:brow_y2, brow_x1:brow_x2] = 255
        rois[side] = roi
    return rois


def detect_face_landmarks(
    image_rgb: np.ndarray,
    detector: vision.FaceLandmarker,
):
    image_rgb = np.ascontiguousarray(image_rgb)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def landmarks_to_xy(landmarks, indices, width: int, height: int) -> np.ndarray:
    return np.array(
        [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in indices],
        dtype=np.int32,
    )


def build_landmark_polygon_mask(
    landmarks,
    image_shape: tuple[int, int],
    expand_px: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = image_shape
    full_mask = np.zeros((height, width), dtype=np.uint8)
    side_masks: dict[str, np.ndarray] = {}
    data = {
        "left": LEFT_EYEBROW_LANDMARKS,
        "right": RIGHT_EYEBROW_LANDMARKS,
    }
    for side, indices in data.items():
        pts = landmarks_to_xy(landmarks, indices, width, height)
        side_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(side_mask, cv2.convexHull(pts), 255)
        if expand_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (expand_px * 2 + 1, expand_px * 2 + 1),
            )
            side_mask = cv2.dilate(side_mask, kernel, iterations=1)
        side_masks[side] = side_mask
        full_mask = np.maximum(full_mask, side_mask)
    return full_mask, side_masks


def get_eye_centers_from_landmarks(
    landmarks,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_eye = landmarks_to_xy(landmarks, LEFT_EYE_LANDMARKS, width, height).astype(np.float32)
    right_eye = landmarks_to_xy(landmarks, RIGHT_EYE_LANDMARKS, width, height).astype(np.float32)
    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    return left_center, right_center


def align_face_crop(
    image_rgb: np.ndarray,
    landmarks,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    left_center, right_center = get_eye_centers_from_landmarks(landmarks, width, height)
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = tuple(((left_center + right_center) / 2.0).tolist())
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inverse = cv2.invertAffineTransform(matrix)
    return aligned, matrix, inverse


def warp_mask(mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    warped = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def warp_parsing(parsing: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    warped = cv2.warpAffine(
        parsing,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def merge_bisenet_with_landmark_mask(
    side_mask: np.ndarray,
    raw_mask: np.ndarray,
    brow_prob: np.ndarray,
    landmark_side_mask: np.ndarray,
    min_area: int,
    prob_threshold: float,
) -> np.ndarray:
    if landmark_side_mask.max() == 0:
        return side_mask

    h, w = landmark_side_mask.shape
    expand_px = max(5, int(round(min(h, w) * 0.012)))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expand_px * 2 + 1, expand_px * 2 + 1),
    )
    landmark_support = cv2.dilate(landmark_side_mask, support_kernel, iterations=1)

    soft_candidates = cv2.bitwise_and(
        raw_mask,
        eyebrow_confidence_mask(brow_prob, max(0.08, prob_threshold - 0.16)),
    )
    support = cv2.bitwise_and(soft_candidates, landmark_support)
    merged = np.maximum(side_mask, support)

    if np.count_nonzero(side_mask) == 0:
        merged = support
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Final fallback: when BiSeNet is still too weak on one side, use the
    # landmark eyebrow polygon itself as a conservative structural prior.
    if np.count_nonzero(merged) == 0:
        merged = landmark_side_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.erode(merged, kernel, iterations=1)

    # Hard geometric guardrail: obvious false positives on cheek, nose, etc.
    # are removed by constraining the final mask to remain near the landmark brow band.
    merged = cv2.bitwise_and(merged, landmark_support)

    merged = connected_component_cleanup(merged, min_area=max(1, min_area // 3))
    return merged


def select_best_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask.max() == 0:
        return mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    selected = np.zeros_like(mask)
    mid_x = w // 2
    for half_name, x_start, x_end in (("left", 0, mid_x), ("right", mid_x, w)):
        best_label = None
        best_score = -1e9
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            component_mask = labels == label
            ys, xs = np.where(component_mask)
            if len(xs) == 0:
                continue
            overlap = np.mean((xs >= x_start) & (xs < x_end))
            if overlap < 0.55:
                continue
            score = component_score(stats[label], tuple(centroids[label]), half_name, w, h)
            if score > best_score:
                best_label = label
                best_score = score
        if best_label is not None:
            selected[labels == best_label] = 255
    return selected


def select_best_component_for_side(
    mask: np.ndarray,
    min_area: int,
    side: str,
) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    selected = np.zeros_like(mask)
    x_start, x_end = (0, w // 2) if side == "left" else (w // 2, w)
    best_label = None
    best_score = -1e9
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        component_mask = labels == label
        ys, xs = np.where(component_mask)
        if len(xs) == 0:
            continue
        overlap = np.mean((xs >= x_start) & (xs < x_end))
        if overlap < 0.45:
            continue
        score = component_score(stats[label], tuple(centroids[label]), side, w, h)
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return selected

    selected[labels == best_label] = 255
    base = stats[best_label]
    bx = base[cv2.CC_STAT_LEFT]
    by = base[cv2.CC_STAT_TOP]
    bw = base[cv2.CC_STAT_WIDTH]
    bh = base[cv2.CC_STAT_HEIGHT]
    base_cx, base_cy = centroids[best_label]

    # Merge adjacent eyebrow fragments that were split by weak confidence
    # or tiny gaps inside the same eyebrow arc.
    for label in range(1, num_labels):
        if label == best_label:
            continue
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max(1, min_area // 3):
            continue
        lx = stats[label, cv2.CC_STAT_LEFT]
        ly = stats[label, cv2.CC_STAT_TOP]
        lw = stats[label, cv2.CC_STAT_WIDTH]
        lh = stats[label, cv2.CC_STAT_HEIGHT]
        lcx, lcy = centroids[label]

        same_band = abs(lcy - base_cy) <= max(bh, lh) * 1.2 + h * 0.025
        close_horizontally = (
            abs(lx - (bx + bw)) <= max(bw, lw) * 0.9 + w * 0.03
            or abs((lx + lw) - bx) <= max(bw, lw) * 0.9 + w * 0.03
            or abs(lcx - base_cx) <= max(bw, lw) * 1.2 + w * 0.05
        )
        similar_height = 0.45 <= (lh / max(bh, 1)) <= 2.2
        overlap = np.mean((((np.where(labels == label)[1]) >= x_start) & ((np.where(labels == label)[1]) < x_end)))
        if same_band and close_horizontally and similar_height and overlap >= 0.30:
            selected[labels == label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=1)
    return selected


def find_side_mask_with_fallback(
    raw_mask: np.ndarray,
    brow_prob: np.ndarray,
    roi_mask: np.ndarray,
    side: str,
    min_area: int,
    prob_threshold: float,
) -> np.ndarray:
    thresholds = [
        prob_threshold,
        max(0.12, prob_threshold - 0.08),
        max(0.08, prob_threshold - 0.14),
    ]
    for threshold in thresholds:
        conf_mask = eyebrow_confidence_mask(brow_prob, threshold)
        mask = cv2.bitwise_and(raw_mask, conf_mask)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = connected_component_cleanup(mask, min_area=max(1, min_area // 2))
        selected = select_best_component_for_side(mask, min_area=max(1, min_area // 2), side=side)
        if np.count_nonzero(selected) > 0:
            return selected
    return np.zeros_like(raw_mask)


def build_face_roi_mask(height: int, width: int) -> np.ndarray:
    roi = np.zeros((height, width), dtype=np.uint8)
    x1 = int(width * 0.10)
    x2 = int(width * 0.90)
    y1 = int(height * 0.12)
    y2 = int(height * 0.48)
    roi[y1:y2, x1:x2] = 255
    return roi


def eyebrow_confidence_mask(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8) * 255


def parsing_to_eyebrow_mask(
    parsing: np.ndarray,
    brow_prob: np.ndarray,
    min_area: int,
    prob_threshold: float,
    roi_mask: np.ndarray | None = None,
    side_rois: dict[str, np.ndarray] | None = None,
    landmark_mask: np.ndarray | None = None,
    landmark_side_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    raw_mask = np.isin(parsing, EYEBROW_LABELS).astype(np.uint8) * 255
    if roi_mask is None:
        roi_mask = build_face_roi_mask(*parsing.shape)
    if landmark_mask is not None:
        roi_mask = np.maximum(roi_mask, landmark_mask)
    if side_rois is None:
        conf_mask = eyebrow_confidence_mask(brow_prob, prob_threshold)
        mask = cv2.bitwise_and(raw_mask, conf_mask)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = connected_component_cleanup(mask, min_area=min_area)
        return select_best_components(mask, min_area=min_area)

    left_mask = find_side_mask_with_fallback(
        raw_mask=raw_mask,
        brow_prob=brow_prob,
        roi_mask=side_rois["left"],
        side="left",
        min_area=min_area,
        prob_threshold=prob_threshold,
    )
    right_mask = find_side_mask_with_fallback(
        raw_mask=raw_mask,
        brow_prob=brow_prob,
        roi_mask=side_rois["right"],
        side="right",
        min_area=min_area,
        prob_threshold=prob_threshold,
    )
    if landmark_side_masks is not None:
        left_mask = merge_bisenet_with_landmark_mask(
            side_mask=left_mask,
            raw_mask=raw_mask,
            brow_prob=brow_prob,
            landmark_side_mask=landmark_side_masks["left"],
            min_area=min_area,
            prob_threshold=prob_threshold,
        )
        right_mask = merge_bisenet_with_landmark_mask(
            side_mask=right_mask,
            raw_mask=raw_mask,
            brow_prob=brow_prob,
            landmark_side_mask=landmark_side_masks["right"],
            min_area=min_area,
            prob_threshold=prob_threshold,
        )
    merged = np.maximum(left_mask, right_mask)
    merged = cv2.bitwise_and(merged, roi_mask)
    return merged


def run_face_crop_parsing_single(
    model: BiSeNet,
    image_rgb: np.ndarray,
    device: torch.device,
    input_size: int,
    face_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
    eye_detectors: list[cv2.CascadeClassifier],
    landmark_detector: vision.FaceLandmarker | None,
    min_area: int,
    prob_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    full_h, full_w = image_rgb.shape[:2]
    face_box = detect_primary_face(image_rgb, face_detector, profile_detector)
    if face_box is None:
        pil_image = Image.fromarray(image_rgb)
        parsing, brow_prob = infer_parsing_map(model, pil_image, device, input_size)
        mask = parsing_to_eyebrow_mask(parsing, brow_prob, min_area, prob_threshold)
        return parsing, mask, None

    crop_box = expand_face_box(face_box, full_w, full_h)
    crop_rgb = crop_array(image_rgb, crop_box)
    aligned_rgb = crop_rgb
    to_aligned = None
    to_crop = None
    landmark_mask = None
    landmark_side_masks = None
    if landmark_detector is not None:
        landmarks = detect_face_landmarks(crop_rgb, landmark_detector)
        if landmarks is not None:
            aligned_rgb, to_aligned, to_crop = align_face_crop(crop_rgb, landmarks)
            landmarks = detect_face_landmarks(aligned_rgb, landmark_detector) or landmarks
            landmark_mask, landmark_side_masks = build_landmark_polygon_mask(
                landmarks,
                aligned_rgb.shape[:2],
                expand_px=2,
            )
    crop_pil = Image.fromarray(aligned_rgb)
    crop_parsing, crop_brow_prob = infer_parsing_map(model, crop_pil, device, input_size)
    eye_boxes = detect_eye_boxes(aligned_rgb, eye_detectors)
    eye_pair = pick_eye_pair(eye_boxes, aligned_rgb.shape[1], aligned_rgb.shape[0])
    roi_mask = build_eye_guided_roi_mask(aligned_rgb.shape[0], aligned_rgb.shape[1], eye_pair)
    side_rois = build_eye_guided_half_rois(aligned_rgb.shape[0], aligned_rgb.shape[1], eye_pair)
    crop_mask = parsing_to_eyebrow_mask(
        crop_parsing,
        crop_brow_prob,
        min_area=min_area,
        prob_threshold=prob_threshold,
        roi_mask=roi_mask,
        side_rois=side_rois,
        landmark_mask=landmark_mask,
        landmark_side_masks=landmark_side_masks,
    )
    if to_crop is not None:
        crop_mask = warp_mask(crop_mask, to_crop, (crop_rgb.shape[1], crop_rgb.shape[0]))
        crop_parsing = warp_parsing(crop_parsing, to_crop, (crop_rgb.shape[1], crop_rgb.shape[0]))
    full_parsing = np.zeros((full_h, full_w), dtype=np.uint8)
    x1, y1, x2, y2 = crop_box
    full_parsing[y1:y2, x1:x2] = crop_parsing
    full_mask = paste_crop_mask(crop_mask, (full_h, full_w), crop_box)
    return full_parsing, full_mask, [x1, y1, x2, y2]


def run_face_crop_parsing(
    model: BiSeNet,
    image_rgb: np.ndarray,
    device: torch.device,
    input_size: int,
    face_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
    eye_detectors: list[cv2.CascadeClassifier],
    landmark_detector: vision.FaceLandmarker | None,
    min_area: int,
    prob_threshold: float,
    angle_step: int,
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    candidates: list[tuple[int, np.ndarray, np.ndarray, list[int] | None]] = []
    step = max(1, min(180, angle_step))
    angles = list(range(0, 360, step))
    if 0 not in angles:
        angles.insert(0, 0)
    for angle in angles:
        if angle % 90 == 0:
            rotated = rotate_image_by_k(image_rgb, (angle // 90) % 4)
            inverse_matrix = None
        else:
            rotated, matrix = rotate_image_by_angle(
                image_rgb,
                angle,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
            )
            inverse_matrix = cv2.invertAffineTransform(matrix)
        parsing_rot, mask_rot, face_bbox = run_face_crop_parsing_single(
            model=model,
            image_rgb=rotated,
            device=device,
            input_size=input_size,
            face_detector=face_detector,
            profile_detector=profile_detector,
            eye_detectors=eye_detectors,
            landmark_detector=landmark_detector,
            min_area=min_area,
            prob_threshold=prob_threshold,
        )
        if angle % 90 == 0:
            parsing = unrotate_image_by_k(parsing_rot, (angle // 90) % 4)
            mask = unrotate_image_by_k(mask_rot, (angle // 90) % 4)
        else:
            parsing = warp_parsing(
                parsing_rot,
                inverse_matrix,
                (image_rgb.shape[1], image_rgb.shape[0]),
            )
            mask = warp_mask(
                mask_rot,
                inverse_matrix,
                (image_rgb.shape[1], image_rgb.shape[0]),
            )
        score = int(np.count_nonzero(mask))
        if face_bbox is not None:
            score += 500
        candidates.append((score, parsing, mask, face_bbox))
        if score > 800:
            break

    best_score, best_parsing, best_mask, best_face_bbox = max(
        candidates,
        key=lambda item: item[0],
    )
    return best_parsing, best_mask, best_face_bbox


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def process_images(
    input_dir: Path,
    output_dir: Path,
    model: BiSeNet,
    device: torch.device,
    landmark_detector: vision.FaceLandmarker | None,
    input_size: int,
    padding_values: list[int],
    min_area: int,
    prob_threshold: float,
    angle_step: int,
) -> list[MaskStats]:
    tight_dir = output_dir / "tight"
    overlay_dir = output_dir / "overlay"
    parsing_dir = output_dir / "parsing_debug"
    extracted_dir = output_dir / "extracted"
    padded_dirs = {pad: output_dir / f"padded_{pad}px" for pad in padding_values}
    face_detector = get_face_detector()
    profile_detector = get_profile_face_detector()
    eye_detectors = get_eye_detectors()
    output_stem_map = build_output_stem_map(input_dir)

    for folder in [tight_dir, overlay_dir, parsing_dir, extracted_dir, *padded_dirs.values()]:
        folder.mkdir(parents=True, exist_ok=True)

    stats: list[MaskStats] = []
    for image_path in iter_images(input_dir):
        try:
            image = load_image(image_path)
        except Exception:
            continue
        image_rgb = np.asarray(image)
        parsing, tight_mask, face_bbox = run_face_crop_parsing(
            model=model,
            image_rgb=image_rgb,
            device=device,
            input_size=input_size,
            face_detector=face_detector,
            profile_detector=profile_detector,
            eye_detectors=eye_detectors,
            landmark_detector=landmark_detector,
            min_area=min_area,
            prob_threshold=prob_threshold,
            angle_step=angle_step,
        )

        stem = output_stem_map.get(image_path, image_path.stem)
        save_mask(tight_dir / f"{stem}_tight_mask.png", tight_mask)
        save_mask(parsing_dir / f"{stem}_parsing.png", parsing)
        Image.fromarray(build_overlay(image_rgb, tight_mask, (255, 80, 80))).save(
            overlay_dir / f"{stem}_tight_overlay.jpg",
            quality=92,
        )
        Image.fromarray(extract_foreground_on_white(image_rgb, tight_mask)).save(
            extracted_dir / f"{stem}_tight_white_bg.png"
        )

        padded_pixel_stats: dict[str, int] = {}
        padded_bbox_stats: dict[str, list[int] | None] = {}
        for pad in padding_values:
            padded_mask = dilate_mask(tight_mask, pad)
            save_mask(padded_dirs[pad] / f"{stem}_padded_{pad}px_mask.png", padded_mask)
            Image.fromarray(build_overlay(image_rgb, padded_mask, (80, 255, 120))).save(
                overlay_dir / f"{stem}_padded_{pad}px_overlay.jpg",
                quality=92,
            )
            padded_pixel_stats[f"{pad}px"] = int(np.count_nonzero(padded_mask))
            padded_bbox_stats[f"{pad}px"] = bbox_from_mask(padded_mask)

        stats.append(
            MaskStats(
                image_name=image_path.name,
                width=image.width,
                height=image.height,
                tight_pixels=int(np.count_nonzero(tight_mask)),
                bbox_xyxy=bbox_from_mask(tight_mask),
                face_bbox_xyxy=face_bbox,
                padded_pixels=padded_pixel_stats,
                padded_bboxes_xyxy=padded_bbox_stats,
            )
        )
    return stats


def write_manifest(output_dir: Path, stats: list[MaskStats], args: argparse.Namespace) -> None:
    payload = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "weights": str(args.weights.resolve()),
        "mediapipe_model": str(args.mediapipe_model.resolve()),
        "input_size": args.input_size,
        "padding_values": args.padding,
        "min_component_area": args.min_component_area,
        "eyebrow_prob_threshold": args.eyebrow_prob_threshold,
        "angle_step": args.angle_step,
        "images": [s.__dict__ for s in stats],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiSeNet 기반 눈썹 마스크 생성기: 정밀 마스크와 1~2px 패딩 마스크를 함께 저장합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eyes"),
        help="입력 이미지 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bisenet_outputs"),
        help="출력 폴더",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights") / "79999_iter.pth",
        help="BiSeNet 가중치 파일 경로",
    )
    parser.add_argument(
        "--mediapipe-model",
        type=Path,
        default=Path("weights") / "face_landmarker.task",
        help="MediaPipe FaceLandmarker 모델 파일 경로",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="BiSeNet 입력 해상도",
    )
    parser.add_argument(
        "--padding",
        type=int,
        nargs="+",
        default=[1, 2],
        help="정밀 마스크에서 확장할 패딩 픽셀 수 목록",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=4,
        help="잔여 노이즈 제거용 최소 연결 컴포넌트 면적",
    )
    parser.add_argument(
        "--eyebrow-prob-threshold",
        type=float,
        default=0.28,
        help="눈썹 클래스 확률이 이 값보다 낮으면 제거",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=30,
        help="multi-angle inference 회전 간격(도). 30이면 0~330도를 30도 단위로 시도",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="추론 장치",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="가중치가 있어도 다시 다운로드",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {args.input_dir}")

    ensure_weights(args.weights, force_download=args.force_download)
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    landmark_detector = create_face_landmarker(args.mediapipe_model)
    stats = process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model,
        device=device,
        landmark_detector=landmark_detector,
        input_size=args.input_size,
        padding_values=sorted(set(p for p in args.padding if p >= 0)),
        min_area=max(1, args.min_component_area),
        prob_threshold=max(0.0, min(1.0, args.eyebrow_prob_threshold)),
        angle_step=max(1, args.angle_step),
    )
    write_manifest(args.output_dir, stats, args)
    print(f"처리 완료: {len(stats)}장")
    print(f"출력 폴더: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
