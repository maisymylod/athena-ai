"""Pull a public real-vs-AI image dataset from HuggingFace and lay it out
under data/raw/{real,synthetic}/ so the existing Trainer can consume it.

Default dataset: Madushan996/real_fake_images — separate `real/` and
`fake_fx/` directories of PNGs, no auth required, ~1000 of each.

Also extracts a small held-out sample for the landing-page "Try with these"
example tiles, so the demo shows the model on test images it has never
seen during training.
"""

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)


def _list_files(api: HfApi, repo: str, prefix: str) -> list[str]:
    info = api.dataset_info(repo)
    return [s.rfilename for s in info.siblings if s.rfilename.startswith(prefix)]


def _download(repo: str, rel: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo, filename=rel, repo_type="dataset", local_dir=None
        )
    )


def _save_thumbnail(src: Path, dst: Path, size: int = 400) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    edge = min(w, h)
    img = img.crop(((w - edge) // 2, (h - edge) // 2, (w + edge) // 2, (h + edge) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    img.save(dst, quality=88, optimize=True)


def _save_training(src: Path, dst: Path, max_edge: int) -> None:
    img = Image.open(src).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    img.save(dst, quality=88, optimize=True)


def prepare(
    repo: str,
    real_prefix: str,
    synth_prefixes: list[str],
    per_synth_prefix_limit: int,
    out_dir: Path,
    examples_dir: Path,
    examples_per_class: int,
    real_limit: int,
    image_size: int,
    workers: int,
) -> None:
    api = HfApi()
    out_real = out_dir / "real"
    out_synth = out_dir / "synthetic"
    out_real.mkdir(parents=True, exist_ok=True)
    out_synth.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing files in %s …", repo)
    real_files = sorted(_list_files(api, repo, real_prefix))[:real_limit]

    # Pull `per_synth_prefix_limit` from each synthetic source — gives the model
    # exposure to multiple generators rather than overfitting to one.
    synth_files: list[str] = []
    for pfx in synth_prefixes:
        items = sorted(_list_files(api, repo, pfx))[:per_synth_prefix_limit]
        logger.info("  %s: %d files", pfx, len(items))
        synth_files.extend(items)

    logger.info("found %d real, %d synthetic across %d generators",
                len(real_files), len(synth_files), len(synth_prefixes))

    if not real_files or not synth_files:
        logger.error(
            "Empty file list for one class — check prefixes. real=%d synth=%d",
            len(real_files), len(synth_files),
        )
        sys.exit(1)

    # First N from each class go to examples_dir (held-out from training)
    held_out: set[str] = set()

    for i, rel in enumerate(real_files[:examples_per_class], start=1):
        held_out.add(rel)
        local = _download(repo, rel)
        _save_thumbnail(local, examples_dir / f"real_{i}.jpg")
        logger.info("example: real_%d.jpg ← %s", i, rel)

    # Pick one example from each synthetic generator if possible
    synth_examples: list[str] = []
    by_prefix: dict[str, list[str]] = {p: [] for p in synth_prefixes}
    for f in synth_files:
        for p in synth_prefixes:
            if f.startswith(p):
                by_prefix[p].append(f)
                break
    rotation = list(synth_prefixes)
    while len(synth_examples) < examples_per_class:
        progressed = False
        for p in rotation:
            if by_prefix[p] and len(synth_examples) < examples_per_class:
                synth_examples.append(by_prefix[p].pop(0))
                progressed = True
        if not progressed:
            break
    for i, rel in enumerate(synth_examples, start=1):
        held_out.add(rel)
        local = _download(repo, rel)
        _save_thumbnail(local, examples_dir / f"ai_{i}.jpg")
        logger.info("example: ai_%d.jpg ← %s", i, rel)

    # The rest go to data/raw/{real,synthetic}/
    def fetch_and_save(rel: str, target_dir: Path) -> tuple[bool, str]:
        if rel in held_out:
            return (False, "held-out")
        try:
            local = _download(repo, rel)
            _save_training(local, target_dir / Path(rel).with_suffix(".jpg").name, image_size)
            return (True, "ok")
        except Exception as exc:
            logger.warning("skip %s: %s", rel, exc)
            return (False, str(exc))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for rel in real_files:
            futures[pool.submit(fetch_and_save, rel, out_real)] = ("real", rel)
        for rel in synth_files:
            futures[pool.submit(fetch_and_save, rel, out_synth)] = ("synthetic", rel)

        real_ok = synth_ok = 0
        for done, fut in enumerate(as_completed(futures), start=1):
            kind, _rel = futures[fut]
            saved, _msg = fut.result()
            if saved:
                if kind == "real":
                    real_ok += 1
                else:
                    synth_ok += 1
            if done % 100 == 0:
                logger.info("progress: %d/%d", done, len(futures))

    logger.info("wrote %d real + %d synthetic to %s", real_ok, synth_ok, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Madushan996/real_fake_images")
    parser.add_argument("--real-prefix", default="real/")
    parser.add_argument(
        "--synth-prefixes",
        default="fake_fx/,fake_sd/,fake_z/",
        help="Comma-separated synthetic-image folder prefixes",
    )
    parser.add_argument(
        "--per-synth-prefix-limit",
        default=400,
        type=int,
        help="Pull at most this many files from each synthetic prefix.",
    )
    parser.add_argument(
        "--real-limit",
        default=1200,
        type=int,
        help="Pull at most this many real files.",
    )
    parser.add_argument("--out-dir", default="data/raw", type=Path)
    parser.add_argument("--examples-dir", default="static/examples", type=Path)
    parser.add_argument("--examples-per-class", default=3, type=int)
    parser.add_argument(
        "--image-size",
        default=320,
        type=int,
        help="Long-edge resize for training images. 0 = keep original.",
    )
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    synth_prefixes = [p.strip() for p in args.synth_prefixes.split(",") if p.strip()]
    prepare(
        args.repo,
        args.real_prefix,
        synth_prefixes,
        args.per_synth_prefix_limit,
        args.out_dir,
        args.examples_dir,
        args.examples_per_class,
        args.real_limit,
        args.image_size,
        args.workers,
    )


if __name__ == "__main__":
    main()
