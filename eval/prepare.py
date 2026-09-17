from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SINTEL_TAG = 202021.25
SINTEL_ARCHIVES = (
    (
        "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
        "MPI-Sintel-training_images.zip",
        ("clean",),
    ),
    (
        "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip",
        "MPI-Sintel-depth-training-20150305.zip",
        ("depth", "camdata_left"),
    ),
)
SINTEL_HF_REPO = "KevinConnorLee/Sintel"
ETH3D_HF_REPO = "Livioni/eth3d_omnivggt"
SINTEL_HF_REVISION = "8304a6a05a71c5099eff2c2fb729858c6b018711"
ETH3D_HF_REVISION = "64154bdca815ef7712f5f2621e231bac7dc99d59"


@dataclass
class Camera:
    model: str
    width: int
    height: int
    parameters: np.ndarray

    def __post_init__(self):
        if self.model == "SIMPLE_PINHOLE":
            self.fx = self.fy = float(self.parameters[0])
            self.cx, self.cy = map(float, self.parameters[1:3])
            self.distortion = np.zeros(0)
        elif self.model in {"PINHOLE", "THIN_PRISM_FISHEYE"}:
            self.fx, self.fy, self.cx, self.cy = map(float, self.parameters[:4])
            self.distortion = self.parameters[4:].astype(np.float64)
        else:
            raise ValueError(f"unsupported COLMAP camera model: {self.model}")

    @property
    def intrinsic(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-omega-eval"})
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        temporary.open("wb") as output,
    ):
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    os.replace(temporary, destination)
    return destination


def _hf_file(repo: str, filename: str, destination: Path, revision: str | None) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            local_dir=str(destination),
        )
    )


def _hf_snapshot(
    repo: str, destination: Path, patterns: list[str], revision: str | None
) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            allow_patterns=patterns,
        )
    )


def _read_colmap_cameras(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        cameras[int(parts[0])] = Camera(
            parts[1],
            int(parts[2]),
            int(parts[3]),
            np.asarray([float(value) for value in parts[4:]]),
        )
    return cameras


def _read_colmap_images(path: Path) -> dict[str, dict]:
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    images = {}
    for line in lines[::2]:
        parts = line.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )
        matrix[:3, 3] = np.asarray([float(value) for value in parts[5:8]])
        images[parts[9]] = {"camera_id": int(parts[8]), "extrinsic": matrix}
    return images


def _project_fisheye(points: np.ndarray, camera: Camera) -> np.ndarray:
    if camera.model != "THIN_PRISM_FISHEYE" or len(camera.distortion) != 8:
        raise ValueError(f"expected THIN_PRISM_FISHEYE, got {camera.model}")
    k1, k2, p1, p2, k3, k4, sx1, sy1 = camera.distortion
    u = points[..., 0] / points[..., 2]
    v = points[..., 1] / points[..., 2]
    radius = np.hypot(u, v)
    scale = np.where(radius > 1e-12, np.arctan(radius) / np.maximum(radius, 1e-12), 1.0)
    u, v = u * scale, v * scale
    u2, v2 = u * u, v * v
    rho2 = u2 + v2
    radial = rho2 * (k1 + rho2 * (k2 + rho2 * (k3 + rho2 * k4)))
    du = u * radial + 2 * p1 * u * v + p2 * (rho2 + 2 * u2) + sx1 * rho2
    dv = v * radial + 2 * p2 * u * v + p1 * (rho2 + 2 * v2) + sy1 * rho2
    return np.stack(
        [camera.fx * (u + du) + camera.cx, camera.fy * (v + dv) + camera.cy],
        axis=-1,
    )


def _undistort_depth(
    raw: np.ndarray, distorted: Camera, undistorted: Camera
) -> np.ndarray:
    x, y = np.meshgrid(
        np.arange(undistorted.width) + 0.5,
        np.arange(undistorted.height) + 0.5,
        indexing="xy",
    )
    rays = np.stack(
        [
            (x - undistorted.cx) / undistorted.fx,
            (y - undistorted.cy) / undistorted.fy,
            np.ones_like(x),
        ],
        axis=-1,
    )
    pixels = _project_fisheye(rays, distorted)
    columns = np.floor(pixels[..., 0]).astype(np.int64)
    rows = np.floor(pixels[..., 1]).astype(np.int64)
    mask = (
        (columns >= 0)
        & (columns < distorted.width)
        & (rows >= 0)
        & (rows < distorted.height)
    )
    output = np.zeros((undistorted.height, undistorted.width), dtype=np.float32)
    output[mask] = raw[rows[mask], columns[mask]]
    return output


def _extract_7z(
    archive: Path, destination: Path, calibration_only: bool = False
) -> None:
    import py7zr

    with py7zr.SevenZipFile(archive) as handle:
        targets = None
        if calibration_only:
            targets = [
                name
                for name in handle.getnames()
                if "/dslr_calibration_jpg/" in name
                and Path(name).name in {"cameras.txt", "images.txt", "points3D.txt"}
            ]
        handle.extract(path=destination, targets=targets)


def _extract_sintel(
    archive: Path, destination: Path, subdirs: tuple[str, ...], scenes: tuple[str, ...]
) -> None:
    targets = {f"training/{subdir}/{scene}/" for subdir in subdirs for scene in scenes}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            normalized = member.filename.replace("\\", "/")
            if member.is_dir() or not any(target in normalized for target in targets):
                continue
            start = normalized.index("training/")
            output = destination / normalized[start:]
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, output.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


def _find_scene(root: Path, scene: str) -> Path:
    direct = root / scene
    if (direct / "dslr_calibration_undistorted").is_dir():
        return direct
    matches = [
        path
        for path in root.glob(f"**/{scene}")
        if (path / "dslr_calibration_undistorted").is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve scene {scene} under {root}")
    return matches[0]


def _eth3d_raw(args: argparse.Namespace, scenes: tuple[str, ...]) -> tuple[Path, str]:
    if args.raw_root:
        return Path(args.raw_root).expanduser().resolve(), "local"
    work = (
        Path(args.work_dir or Path(args.output).with_name("eth3d_downloads"))
        .expanduser()
        .resolve()
    )
    raw = work / "raw"
    if args.source in {"auto", "official", "local"}:
        try:
            archives = (
                Path(args.archive_dir).expanduser().resolve()
                if args.archive_dir
                else work / "archives"
            )
            archives.mkdir(parents=True, exist_ok=True)
            names = [
                "multi_view_training_dslr_undistorted.7z",
                "multi_view_training_dslr_jpg.7z",
                *(f"{scene}_dslr_depth.7z" for scene in scenes),
            ]
            for name in names:
                archive = archives / name
                if not archive.is_file():
                    if args.source == "local":
                        raise FileNotFoundError(archive)
                    archive = _download(f"https://www.eth3d.net/data/{name}", archive)
                print(f"{archive.name} sha256={_sha256(archive)}")
                _extract_7z(archive, raw, calibration_only=name.endswith("dslr_jpg.7z"))
            return raw, "official"
        except Exception as error:
            if args.source != "auto":
                raise
            print(f"official ETH3D download unavailable: {error}")
    patterns = []
    for scene in scenes:
        patterns.extend(
            [
                f"{scene}/dslr_calibration_jpg/*",
                f"{scene}/dslr_calibration_undistorted/*",
                f"{scene}/images/dslr_images_undistorted/*",
                f"{scene}/ground_truth_depth/dslr_images/*",
            ]
        )
    repo = args.hf_repo or ETH3D_HF_REPO
    revision = args.hf_revision or (
        ETH3D_HF_REVISION if repo == ETH3D_HF_REPO else None
    )
    return _hf_snapshot(repo, raw, patterns, revision), "huggingface"


def _prepare_eth3d(args: argparse.Namespace) -> None:
    frame_map = json.loads((ROOT / "frames" / "eth3d.json").read_text())
    scenes = tuple(args.scenes or frame_map)
    raw_root, source = _eth3d_raw(args, scenes)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene in scenes:
        scene_root = _find_scene(raw_root, scene)
        distorted_root = scene_root / "dslr_calibration_jpg"
        undistorted_root = scene_root / "dslr_calibration_undistorted"
        distorted_cameras = _read_colmap_cameras(distorted_root / "cameras.txt")
        distorted_images = _read_colmap_images(distorted_root / "images.txt")
        undistorted_cameras = _read_colmap_cameras(undistorted_root / "cameras.txt")
        undistorted_images = _read_colmap_images(undistorted_root / "images.txt")
        records = []
        for name, item in sorted(
            undistorted_images.items(), key=lambda pair: Path(pair[0]).stem
        ):
            stem = Path(name).stem
            distorted_name = f"dslr_images/{stem}.JPG"
            if distorted_name not in distorted_images:
                continue
            camera = undistorted_cameras[item["camera_id"]]
            records.append((stem, item, camera, distorted_images[distorted_name]))
        selected = set(frame_map[scene])
        scene_output = output_root / scene
        (scene_output / "images").mkdir(parents=True, exist_ok=True)
        (scene_output / "depths").mkdir(parents=True, exist_ok=True)
        names, extrinsics, intrinsics = [], [], []
        for index, (stem, item, camera, distorted_item) in enumerate(records):
            filename = f"{stem}.JPG"
            names.append(filename)
            extrinsics.append(item["extrinsic"][:3])
            intrinsics.append(camera.intrinsic)
            if index not in selected:
                continue
            image_source = scene_root / "images" / "dslr_images_undistorted" / filename
            shutil.copy2(image_source, scene_output / "images" / filename)
            distorted_camera = distorted_cameras[distorted_item["camera_id"]]
            raw_path = scene_root / "ground_truth_depth" / "dslr_images" / filename
            raw_depth = np.fromfile(raw_path, dtype=np.float32)
            expected = distorted_camera.width * distorted_camera.height
            if raw_depth.size != expected:
                raise ValueError(f"invalid ETH3D depth file: {raw_path}")
            raw_depth = np.nan_to_num(
                raw_depth.reshape(distorted_camera.height, distorted_camera.width),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            depth = _undistort_depth(raw_depth, distorted_camera, camera)
            np.savez_compressed(scene_output / "depths" / f"{stem}.npz", depth=depth)
            total += 1
        (scene_output / "frames.txt").write_text("\n".join(names) + "\n")
        np.savez_compressed(
            scene_output / "cameras.npz",
            extrinsics=np.asarray(extrinsics),
            intrinsics=np.asarray(intrinsics),
        )
        print(f"{scene}: {len(selected)} frames")
    (output_root / "source.json").write_text(
        json.dumps(
            {
                "dataset": "ETH3D",
                "frames": total,
                "location": (
                    (args.hf_repo or ETH3D_HF_REPO)
                    if source == "huggingface"
                    else (
                        "https://www.eth3d.net/data/"
                        if source == "official"
                        else "local"
                    )
                ),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "revision": (
                    args.hf_revision
                    or (
                        ETH3D_HF_REVISION
                        if (args.hf_repo or ETH3D_HF_REPO) == ETH3D_HF_REPO
                        else None
                    )
                )
                if source == "huggingface"
                else None,
                "source": source,
            },
            indent=2,
        )
        + "\n"
    )


def _read_sintel_depth(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if np.fromfile(handle, np.float32, 1)[0] != SINTEL_TAG:
            raise ValueError(path)
        width = int(np.fromfile(handle, np.int32, 1)[0])
        height = int(np.fromfile(handle, np.int32, 1)[0])
        values = np.fromfile(handle, np.float32)
    if values.size != width * height:
        raise ValueError(path)
    return values.reshape(height, width)


def _read_sintel_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        if np.fromfile(handle, np.float32, 1)[0] != SINTEL_TAG:
            raise ValueError(path)
        values = np.fromfile(handle, np.float64, 21)
    if values.size != 21:
        raise ValueError(path)
    return values[:9].reshape(3, 3), values[9:].reshape(3, 4)


def _sintel_raw(args: argparse.Namespace, scenes: tuple[str, ...]) -> tuple[Path, str]:
    if args.raw_root:
        return Path(args.raw_root).expanduser().resolve(), "local"
    work = (
        Path(args.work_dir or Path(args.output).with_name("sintel_downloads"))
        .expanduser()
        .resolve()
    )
    raw = work / "raw"
    if args.source in {"auto", "official", "local"}:
        try:
            archives = (
                Path(args.archive_dir).expanduser().resolve()
                if args.archive_dir
                else work / "archives"
            )
            archives.mkdir(parents=True, exist_ok=True)
            for official_url, mirror_name, subdirs in SINTEL_ARCHIVES:
                official_name = official_url.rsplit("/", 1)[-1]
                archive = next(
                    (
                        path
                        for path in (archives / official_name, archives / mirror_name)
                        if path.is_file()
                    ),
                    None,
                )
                if archive is None:
                    if args.source == "local":
                        raise FileNotFoundError(archives / official_name)
                    archive = _download(official_url, archives / official_name)
                print(f"{archive.name} sha256={_sha256(archive)}")
                _extract_sintel(archive, raw, subdirs, scenes)
            return raw, "official"
        except Exception as error:
            if args.source != "auto":
                raise
            print(f"official Sintel download unavailable: {error}")
    archives = work / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for _, mirror_name, subdirs in SINTEL_ARCHIVES:
        repo = args.hf_repo or SINTEL_HF_REPO
        revision = args.hf_revision or (
            SINTEL_HF_REVISION if repo == SINTEL_HF_REPO else None
        )
        archive = _hf_file(repo, mirror_name, archives, revision)
        print(f"{archive.name} sha256={_sha256(archive)}")
        _extract_sintel(archive, raw, subdirs, scenes)
    return raw, "huggingface"


def _prepare_sintel(args: argparse.Namespace) -> None:
    frame_map = json.loads((ROOT / "frames" / "sintel.json").read_text())
    scenes = tuple(args.scenes or frame_map)
    raw_root, source = _sintel_raw(args, scenes)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene in scenes:
        image_root = raw_root / "training" / "clean" / scene
        depth_root = raw_root / "training" / "depth" / scene
        camera_root = raw_root / "training" / "camdata_left" / scene
        images = sorted(image_root.glob("frame_*.png"))
        selected = set(frame_map[scene])
        scene_output = output_root / scene
        (scene_output / "images").mkdir(parents=True, exist_ok=True)
        (scene_output / "depths").mkdir(parents=True, exist_ok=True)
        names, extrinsics, intrinsics = [], [], []
        for index, image_path in enumerate(images):
            camera_path = camera_root / f"frame_{index + 1:04d}.cam"
            depth_path = depth_root / f"frame_{index + 1:04d}.dpt"
            intrinsic, extrinsic = _read_sintel_camera(camera_path)
            names.append(f"{index:05d}.png")
            height, width = 360, 846
            scaled = intrinsic.copy()
            scaled[0] *= width / 1024
            scaled[1] *= height / 436
            intrinsics.append(scaled)
            extrinsics.append(extrinsic)
            if index not in selected:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            resized = cv2.resize(
                image.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                str(scene_output / "images" / f"{index:05d}.png"),
                np.clip(resized, 0, 255).astype(np.uint8),
            )
            depth = cv2.resize(
                _read_sintel_depth(depth_path),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            np.savez_compressed(
                scene_output / "depths" / f"{index:05d}.npz", depth=depth
            )
            total += 1
        (scene_output / "frames.txt").write_text("\n".join(names) + "\n")
        np.savez_compressed(
            scene_output / "cameras.npz",
            extrinsics=np.asarray(extrinsics),
            intrinsics=np.asarray(intrinsics),
        )
        print(f"{scene}: {len(selected)} frames")
    (output_root / "source.json").write_text(
        json.dumps(
            {
                "dataset": "MPI-Sintel",
                "frames": total,
                "location": (
                    (args.hf_repo or SINTEL_HF_REPO)
                    if source == "huggingface"
                    else (
                        [item[0] for item in SINTEL_ARCHIVES]
                        if source == "official"
                        else "local"
                    )
                ),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "revision": (
                    args.hf_revision
                    or (
                        SINTEL_HF_REVISION
                        if (args.hf_repo or SINTEL_HF_REPO) == SINTEL_HF_REPO
                        else None
                    )
                )
                if source == "huggingface"
                else None,
                "source": source,
            },
            indent=2,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("eth3d", "sintel"):
        command = subparsers.add_parser(dataset)
        command.add_argument("--output", required=True)
        command.add_argument(
            "--source", choices=("auto", "official", "hf", "local"), default="auto"
        )
        command.add_argument("--raw-root")
        command.add_argument("--archive-dir")
        command.add_argument("--work-dir")
        command.add_argument("--hf-repo")
        command.add_argument("--hf-revision")
        command.add_argument("--scenes", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset == "eth3d":
        _prepare_eth3d(args)
    else:
        _prepare_sintel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
