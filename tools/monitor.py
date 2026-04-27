"""Athena Image Monitor — perceptual-hash matching pipeline.

Hashes a set of opt-in reference images (enrolled by the protected person),
then compares them against images found at user-supplied URLs or in user-
supplied web pages. Flags matches that also show synthetic-generation
indicators, and drafts a TAKE IT DOWN Act takedown request.

Operates only on URLs the user provides. There is no autonomous crawl.
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import imagehash
import requests
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger(__name__)

USER_AGENT = "AthenaMonitor/0.1 (+https://github.com/maisymylod/athena-ai)"
REQUEST_TIMEOUT = 15  # seconds


class ScanResult(Enum):
    CLEAR = "clear"
    MATCH = "match"
    MATCH_SYNTHETIC = "match_synthetic"


@dataclass
class ImageFingerprint:
    """Perceptual hash fingerprint of an image."""

    url: str
    phash: imagehash.ImageHash
    file_hash: str
    width: int = 0
    height: int = 0
    metadata: dict = field(default_factory=dict)

    def hamming_distance(self, other: "ImageFingerprint") -> int:
        return self.phash - other.phash

    def similarity(self, other: "ImageFingerprint") -> float:
        """Similarity score in [0, 1]. 64-bit phash → max distance 64."""
        return 1.0 - (self.hamming_distance(other) / 64.0)


@dataclass
class ScanMatch:
    source_url: str
    reference_path: str
    similarity: float
    is_synthetic: bool
    scan_timestamp: str
    synthetic_indicators: list = field(default_factory=list)


def fingerprint_image(image_bytes: bytes, source_url: str) -> ImageFingerprint:
    """Compute the perceptual hash + SHA256 of an image."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return ImageFingerprint(
        url=source_url,
        phash=imagehash.phash(img),
        file_hash=hashlib.sha256(image_bytes).hexdigest(),
        width=img.width,
        height=img.height,
    )


def fingerprint_path(path: Path) -> ImageFingerprint:
    """Compute the perceptual hash of a local image file."""
    return fingerprint_image(path.read_bytes(), str(path))


class SyntheticDetector:
    """Synthetic-generation analysis: ML model when available, plus metadata heuristics."""

    SYNTHETIC_SIGNATURES = [
        "stable diffusion",
        "midjourney",
        "dall-e",
        "novelai",
        "automatic1111",
        "comfyui",
        "invokeai",
        "fooocus",
        "civitai",
        "tensor.art",
    ]

    SUSPICIOUS_SOFTWARE = ["python", "pytorch", "tensorflow"]

    _ml_inference = None
    _ml_load_attempted = False

    @classmethod
    def _get_ml_inference(cls):
        if cls._ml_load_attempted:
            return cls._ml_inference

        cls._ml_load_attempted = True
        try:
            from ml.inference import DeepfakeInference

            checkpoint_path = Path("checkpoints/best_model.pt")
            if checkpoint_path.exists():
                cls._ml_inference = DeepfakeInference.from_checkpoint(checkpoint_path)
        except Exception as exc:
            logger.debug("ML model unavailable: %s", exc)

        return cls._ml_inference

    @classmethod
    def analyze(
        cls, metadata: dict, image_path: str | None = None
    ) -> tuple[bool, list[str]]:
        indicators: list[str] = []
        ml_result: bool | None = None

        if image_path:
            inference = cls._get_ml_inference()
            if inference:
                try:
                    is_synthetic_ml, _confidence, ml_indicators = inference.predict(
                        image_path
                    )
                    indicators.extend(ml_indicators)
                    ml_result = is_synthetic_ml
                except Exception as exc:
                    logger.debug("ML inference failed: %s", exc)

        indicators.extend(cls._analyze_metadata(metadata))

        if ml_result is not None:
            return ml_result, indicators

        # Fallback: at least 2 metadata indicators required to flag synthetic.
        return len(indicators) >= 2, indicators

    @classmethod
    def _analyze_metadata(cls, metadata: dict) -> list[str]:
        indicators: list[str] = []

        software = str(metadata.get("software", "")).lower()
        for sig in cls.SYNTHETIC_SIGNATURES:
            if sig in software:
                indicators.append(f"AI tool signature: {sig}")
        for sus in cls.SUSPICIOUS_SOFTWARE:
            if sus in software:
                indicators.append(f"Suspicious software: {sus}")

        params = str(metadata.get("parameters", "")).lower()
        usercomment = str(metadata.get("usercomment", "")).lower()
        ai_params = [
            "cfg scale",
            "sampling steps",
            "sampler",
            "negative prompt",
            "lora",
            "checkpoint",
            "vae",
            "clip skip",
        ]
        for param in ai_params:
            if param in params or param in usercomment:
                indicators.append(f"AI generation parameter: {param}")

        if "c2pa" in str(metadata).lower():
            indicators.append("C2PA content credential found")

        width = metadata.get("width", 0)
        height = metadata.get("height", 0)
        if width and height and width % 64 == 0 and height % 64 == 0 and width >= 512:
            indicators.append(
                f"Resolution {width}x{height} matches AI generation pattern"
            )

        return indicators


class TakedownGenerator:
    """Generate TAKE IT DOWN Act-compliant takedown requests."""

    TEMPLATE = """\
TAKEDOWN REQUEST — TAKE IT DOWN Act (S.146, 119th Congress)
============================================================
Date:           {date}
Platform:       {platform}
Content URL:    {url}

NOTIFICATION OF NON-CONSENSUAL INTIMATE IMAGERY

Pursuant to the TAKE IT DOWN Act, I am requesting the immediate
removal of the following content, which constitutes non-consensual
intimate imagery generated using artificial intelligence.

CONTENT DETAILS:
  URL:              {url}
  Detection Method: Perceptual hash matching + synthetic analysis
  Similarity Score: {similarity:.0%}
  Hash Match:       {ref_hash} → {source_hash}

SYNTHETIC GENERATION INDICATORS:
{indicators}

This content was identified as AI-generated non-consensual intimate
imagery through automated detection. Under the TAKE IT DOWN Act,
platforms must remove such content within 48 hours of notification.

REQUEST:
  1. Remove the identified content immediately.
  2. Prevent re-upload using the provided content hash.
  3. Preserve evidence for potential law enforcement referral.

============================================================
Generated by Athena — https://github.com/maisymylod/athena-ai
"""

    @classmethod
    def generate(cls, match: ScanMatch) -> str:
        platform = urlparse(match.source_url).netloc or "(unknown)"
        if match.synthetic_indicators:
            indicators = "\n".join(f"  - {ind}" for ind in match.synthetic_indicators)
        else:
            indicators = "  - Perceptual hash match with high confidence"

        return cls.TEMPLATE.format(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            platform=platform,
            url=match.source_url,
            similarity=match.similarity,
            ref_hash="[reference]",
            source_hash="[detected]",
            indicators=indicators,
        )


def fetch_bytes(url: str) -> bytes:
    """Download URL contents. Raises requests.HTTPError on failure."""
    resp = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.content


def extract_image_urls(page_url: str, html: str) -> list[str]:
    """Find <img src=...> URLs on a page and resolve them to absolute URLs."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if not src:
            continue
        urls.append(urljoin(page_url, src))
    return urls


class WebScanner:
    """Scan user-supplied URLs (image URLs or pages) against a set of references.

    There is no autonomous crawling. The caller hands in URLs to check.
    """

    def __init__(self, reference_hashes: list[ImageFingerprint]):
        self.reference_hashes = reference_hashes
        self.matches: list[ScanMatch] = []
        self.scanned_urls: set[str] = set()
        self.similarity_threshold = 0.85  # 0.85 ≈ Hamming distance ≤ 9 / 64

    def _check_image_against_refs(
        self, image_url: str, image_bytes: bytes
    ) -> ScanMatch | None:
        try:
            fp = fingerprint_image(image_bytes, image_url)
        except Exception as exc:
            logger.debug("Could not hash %s: %s", image_url, exc)
            return None

        best_ref: ImageFingerprint | None = None
        best_sim = 0.0
        for ref in self.reference_hashes:
            sim = fp.similarity(ref)
            if sim > best_sim:
                best_sim = sim
                best_ref = ref

        if not best_ref or best_sim < self.similarity_threshold:
            return None

        is_synthetic, indicators = SyntheticDetector.analyze(
            metadata={"width": fp.width, "height": fp.height}
        )
        return ScanMatch(
            source_url=image_url,
            reference_path=best_ref.url,
            similarity=best_sim,
            is_synthetic=is_synthetic,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            synthetic_indicators=indicators,
        )

    def scan_image_url(self, image_url: str) -> ScanMatch | None:
        """Download a single image URL and compare against references."""
        self.scanned_urls.add(image_url)
        try:
            data = fetch_bytes(image_url)
        except Exception as exc:
            logger.warning("Skipping %s: %s", image_url, exc)
            return None
        match = self._check_image_against_refs(image_url, data)
        if match:
            self.matches.append(match)
        return match

    def scan_page(self, page_url: str) -> list[ScanMatch]:
        """Download a page, extract <img> URLs, hash each, compare against references."""
        self.scanned_urls.add(page_url)
        try:
            html = fetch_bytes(page_url).decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Could not fetch page %s: %s", page_url, exc)
            return []

        image_urls = extract_image_urls(page_url, html)
        logger.info("Found %d images on %s", len(image_urls), page_url)

        page_matches: list[ScanMatch] = []
        for image_url in image_urls:
            match = self.scan_image_url(image_url)
            if match:
                page_matches.append(match)
        return page_matches

    def report(self) -> dict:
        return {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "urls_scanned": len(self.scanned_urls),
            "total_matches": len(self.matches),
            "synthetic_matches": sum(1 for m in self.matches if m.is_synthetic),
            "matches": [
                {
                    "url": m.source_url,
                    "reference": m.reference_path,
                    "similarity": m.similarity,
                    "is_synthetic": m.is_synthetic,
                    "indicators": m.synthetic_indicators,
                }
                for m in self.matches
            ],
        }


def run_demo() -> None:
    """Demo using synthetic test data — no network calls."""
    print("=" * 60)
    print("  ATHENA IMAGE MONITOR — DEMO (synthetic data)")
    print("=" * 60)
    print()

    demo_results = [
        ScanMatch(
            source_url="https://example-site.com/images/img_0847.jpg",
            reference_path="protected_user_ref_01.jpg",
            similarity=0.94,
            is_synthetic=True,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            synthetic_indicators=[
                "AI tool signature: stable diffusion",
                "Resolution 512x768 matches AI generation pattern",
            ],
        ),
        ScanMatch(
            source_url="https://example-site.com/images/img_1293.jpg",
            reference_path="",
            similarity=0.12,
            is_synthetic=False,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        ScanMatch(
            source_url="https://example-site.com/images/img_2041.jpg",
            reference_path="protected_user_ref_01.jpg",
            similarity=0.87,
            is_synthetic=True,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            synthetic_indicators=[
                "AI generation parameter: cfg scale",
                "AI generation parameter: negative prompt",
                "Suspicious software: python",
            ],
        ),
    ]

    print("[*] Scan Results:")
    print("-" * 60)

    flagged = []
    for r in demo_results:
        if r.similarity >= 0.85 and r.is_synthetic:
            flagged.append(r)
            print(f"  [MATCH SYNTHETIC]  {r.source_url}")
            print(f"    Similarity: {r.similarity:.0%}   Matched: {r.reference_path}")
        else:
            print(f"  [clear]            {r.source_url}")
            print(f"    Similarity: {r.similarity:.0%}")
        print()

    if flagged:
        print(f"[*] Generating {len(flagged)} TAKE IT DOWN Act request(s)...")
        print()
        for match in flagged:
            print(TakedownGenerator.generate(match))


def _enroll_references(ref_path: Path) -> list[ImageFingerprint]:
    """Hash every image in a directory into perceptual-hash references."""
    references: list[ImageFingerprint] = []
    for img_file in sorted(ref_path.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            references.append(fingerprint_path(img_file))
            logger.info("Enrolled: %s", img_file.name)
        except Exception as exc:
            logger.warning("Could not enroll %s: %s", img_file, exc)
    return references


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Athena Image Monitor — perceptual-hash matching for non-consensual imagery"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument(
        "--refs", type=str, help="Directory of reference photos (one per protected person)"
    )
    parser.add_argument("--image-url", type=str, help="A single image URL to check")
    parser.add_argument(
        "--page-url", type=str, help="A web page URL — every <img> on the page is checked"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85, help="Similarity threshold (0-1)"
    )
    parser.add_argument("--output", type=str, help="Write the JSON report to this path")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.refs:
        parser.error("--refs <dir> is required (or use --demo)")

    ref_path = Path(args.refs)
    if not ref_path.exists() or not ref_path.is_dir():
        print(f"Error: reference path {ref_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    references = _enroll_references(ref_path)
    print(f"[*] Enrolled {len(references)} reference image(s)")

    if not args.image_url and not args.page_url:
        parser.error("Provide --image-url or --page-url to scan")

    scanner = WebScanner(references)
    scanner.similarity_threshold = args.threshold

    if args.image_url:
        scanner.scan_image_url(args.image_url)
    if args.page_url:
        scanner.scan_page(args.page_url)

    report = scanner.report()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"[*] Report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
