#!/usr/bin/env python3
"""
extract_dino_features.py

Frozen self-supervised (DINO) feature extraction for FreeGrain / ImageNet-3L-F.

Reads the repo's own manifests (data/imagenet-F-train.txt, data/imagenet-F-val.txt)
instead of assuming an ImageFolder train/val directory layout, because the
Kingston extraction is flat per-synset (images/<wnid>/<file>.JPEG) and the
manifest is the actual source of truth for the train/val split.

Manifest line format (confirmed from FreeGrainLearning/deit/dataset/imagenet_f.py):
    train: "train/<wnid>/<file>.JPEG <basic> <subord> <class> <fine_label>"
    val:   "val/<wnid>/<file>.JPEG <basic> <subord> <class>"
The leading "train/" or "val/" path component is dropped and replaced with
--images-dir when resolving the file on disk.

- Backbone: DINO ResNet50 (self-supervised, NO label supervision) -> 2048-d.
- Eval transform only, no augmentation -> deterministic, cacheable features.
- Resumable: writes into a .npy-backed memmap and checkpoints progress, so an
  interrupted run picks up where it stopped instead of recomputing everything.

Framing note for the paper: DINO ResNet50 is pretrained self-supervised on
ImageNet-1k *images* (no labels). That removes *label* leakage, but the
features are still in-distribution w.r.t. ImageNet. The honest description is
a "frozen self-supervised representation protocol" -- not a from-scratch
backbone and not an off-domain one. Worth stating explicitly in the paper.

Downstream loading:
    X = np.load("out/train_features.npy", mmap_mode="r")   # (N, 2048)
    L = np.load("out/train_labels.npy")     # (N, 4): basic, subord, class, fine
                                             # val has no fine column -> filled with -1
    meta = json.load(open("out/train_meta.json"))          # manifest path, columns, files
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Tolerate the occasional truncated JPEG instead of crashing a multi-hour run.
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 2048  # ResNet50 penultimate (post global avg-pool)
LABEL_COLS = ["basic", "subord", "class", "fine"]  # fine == -1 for val


class ManifestImageDataset(Dataset):
    """Reads a FreeGrainLearning imagenet-F-{train,val}.txt manifest.

    Resolves each entry against --images-dir (the flat per-synset extraction),
    not against the "train/"/"val/" prefix baked into the manifest, since that
    prefix does not exist as a real directory on disk.
    """

    def __init__(self, manifest_path, images_dir, transform, is_train):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_train = is_train
        self.rel_paths, self.basic, self.subord, self.cls, self.fine = [], [], [], [], []

        with open(manifest_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                p = Path(parts[0])
                rel = Path(*p.parts[1:]) if len(p.parts) > 1 else p  # drop train/ or val/
                self.rel_paths.append(rel)
                self.basic.append(int(parts[1]))
                self.subord.append(int(parts[2]))
                self.cls.append(int(parts[3]))
                self.fine.append(int(parts[4]) if is_train else -1)

    def __len__(self):
        return len(self.rel_paths)

    def resolved_path(self, idx):
        return self.images_dir / self.rel_paths[idx]

    def __getitem__(self, idx):
        path = self.resolved_path(idx)
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, idx


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(arch, device, weights=None):
    """DINO ResNet50 backbone -> 2048-d embeddings, frozen and in eval mode."""
    if weights:
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("teacher", state.get("student", state))
        cleaned = {}
        for k, v in state.items():
            k2 = k.replace("module.", "").replace("backbone.", "")
            if k2.startswith("head") or k2.startswith("fc"):
                continue
            cleaned[k2] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[model] local weights loaded "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        model = torch.hub.load("facebookresearch/dino:main", arch)

    if hasattr(model, "fc") and not isinstance(model.fc, nn.Identity):
        model.fc = nn.Identity()

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def probe_manifest_files(dataset, n=5):
    """Sample a few entries (start/middle/end) and check they exist on disk.

    Catches the "manifest points at a split that was never downloaded/extracted"
    case (e.g. val) before burning hours of compute on it.
    """
    N = len(dataset)
    if N == 0:
        return [], []
    idxs = sorted(set([0, N // 4, N // 2, (3 * N) // 4, N - 1][:n]))
    missing = [(i, str(dataset.resolved_path(i))) for i in idxs
               if not dataset.resolved_path(i).exists()]
    return idxs, missing


@torch.no_grad()
def extract_split(split_name, manifest_path, images_dir, out_dir, model, device,
                  batch_size, num_workers, storage_dtype, checkpoint_every,
                  amp=False, expected_count=None, is_train=True):
    manifest_path = Path(manifest_path)
    assert manifest_path.is_file(), f"manifest not found: {manifest_path}"

    dataset = ManifestImageDataset(manifest_path, images_dir, build_transform(), is_train)
    N = len(dataset)
    print(f"[{split_name}] manifest={manifest_path.name}  {N} entries")
    if expected_count is not None and N != expected_count:
        print(f"[{split_name}] WARNING: expected {expected_count} entries, "
              f"manifest has {N}. Check you're reading the right file.")

    probe_idxs, missing = probe_manifest_files(dataset)
    if missing:
        print(f"[{split_name}] ABORTING: {len(missing)}/{len(probe_idxs)} sampled "
              f"files do not exist under {images_dir}, e.g.:")
        for i, p in missing[:5]:
            print(f"    [{i}] {p}")
        print(f"[{split_name}] This split's images likely haven't been "
              f"downloaded/extracted to disk yet. Skipping '{split_name}' "
              f"rather than failing mid-run.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{split_name}_features.npy"
    lbl_path = out_dir / f"{split_name}_labels.npy"
    meta_path = out_dir / f"{split_name}_meta.json"
    prog_path = out_dir / f"{split_name}_progress.json"

    labels = np.array(list(zip(dataset.basic, dataset.subord, dataset.cls, dataset.fine)),
                      dtype=np.int64)
    np.save(lbl_path, labels)
    meta = {
        "split": split_name,
        "num_images": N,
        "feature_dim": FEATURE_DIM,
        "storage_dtype": storage_dtype,
        "label_columns": LABEL_COLS,
        "manifest": str(manifest_path),
        "images_dir": str(images_dir),
        "files": [str(dataset.rel_paths[i]) for i in range(N)],
    }
    meta_path.write_text(json.dumps(meta))

    np_dtype = np.dtype(storage_dtype)
    if feat_path.exists() and prog_path.exists():
        X = np.lib.format.open_memmap(feat_path, mode="r+")
        if X.shape != (N, FEATURE_DIM) or X.dtype != np_dtype:
            raise SystemExit(
                f"[{split_name}] existing {feat_path.name} is {X.shape}/{X.dtype}, "
                f"expected {(N, FEATURE_DIM)}/{np_dtype}. Delete it to recompute.")
        done = json.loads(prog_path.read_text()).get("done", 0)
    else:
        X = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=np_dtype, shape=(N, FEATURE_DIM))
        done = 0
        prog_path.write_text(json.dumps({"done": 0}))

    if done >= N:
        print(f"[{split_name}] already complete ({done}/{N}). Skipping.")
        return feat_path

    print(f"[{split_name}] resuming at {done}/{N}")
    loader = DataLoader(
        Subset(dataset, range(done, N)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if amp and device.type == "cuda" else nullcontext())

    offset = done
    t0, seen = time.time(), 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast:
            feats = model(imgs)                       # (B, 2048)
        feats = feats.float().cpu().numpy().astype(np_dtype, copy=False)
        nb = feats.shape[0]
        X[offset:offset + nb] = feats
        offset += nb
        seen += nb
        if offset % checkpoint_every < batch_size or offset >= N:
            X.flush()
            prog_path.write_text(json.dumps({"done": offset}))
            rate = seen / max(time.time() - t0, 1e-6)
            eta_min = (N - offset) / max(rate, 1e-6) / 60
            print(f"[{split_name}] {offset}/{N}  "
                  f"{rate:6.1f} img/s  ETA {eta_min:6.1f} min", flush=True)

    X.flush()
    prog_path.write_text(json.dumps({"done": N}))
    print(f"[{split_name}] done -> {feat_path}")
    return feat_path


def main():
    ap = argparse.ArgumentParser(
        description="Frozen DINO ResNet50 feature extraction over FreeGrainLearning manifests.")
    ap.add_argument("--repo-root", type=str, default=str(Path.home() / "Downloads" / "FreeGrainLearning"),
                    help="FreeGrainLearning repo checkout (for default manifest locations)")
    ap.add_argument("--train-manifest", type=str, default=None,
                    help="default: <repo-root>/data/imagenet-F-train.txt")
    ap.add_argument("--val-manifest", type=str, default=None,
                    help="default: <repo-root>/data/imagenet-F-val.txt")
    ap.add_argument("--images-dir", type=str, required=True,
                    help="flat per-synset extraction root, e.g. /Volumes/KINGSTON/imagenet/images")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--arch", default="dino_resnet50")
    ap.add_argument("--weights", default=None,
                    help="optional local DINO .pth if torch.hub is offline")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int,
                    default=min(8, os.cpu_count() or 2))
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast on CUDA (no-op on MPS/CPU)")
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--expected-train", type=int, default=645840)
    ap.add_argument("--expected-val", type=int, default=25250)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    train_manifest = args.train_manifest or (repo_root / "data" / "imagenet-F-train.txt")
    val_manifest = args.val_manifest or (repo_root / "data" / "imagenet-F-val.txt")

    device = get_device(args.device)
    print(f"[device] {device} | batch={args.batch_size} "
          f"workers={args.num_workers} dtype={args.dtype} amp={args.amp}")
    model = load_model(args.arch, device, args.weights)

    jobs = []
    if args.split in ("train", "both"):
        jobs.append(("train", train_manifest, args.expected_train, True))
    if args.split in ("val", "both"):
        jobs.append(("val", val_manifest, args.expected_val, False))

    results = {}
    for name, manifest, exp, is_train in jobs:
        results[name] = extract_split(
            name, manifest, args.images_dir, args.output_dir, model, device,
            args.batch_size, args.num_workers, args.dtype,
            args.checkpoint_every, amp=args.amp, expected_count=exp, is_train=is_train)

    skipped = [name for name, r in results.items() if r is None]
    if skipped:
        print(f"\n[summary] Skipped (images not on disk yet): {', '.join(skipped)}. "
              f"Re-run with --split {'/'.join(skipped)} once available.")
    done = [name for name, r in results.items() if r is not None]
    if done:
        print(f"[summary] Completed: {', '.join(done)}")


if __name__ == "__main__":
    main()
