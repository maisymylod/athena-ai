"""Athena Image Monitor — Web scanning pipeline for detecting
non-consensual AI-generated imagery.

Crawls web pages, computes perceptual hashes, compares against
protected reference images, and generates takedown requests.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse


class ScanResult(Enum):
    CLEAR = "clear"
    MATCH = "match"
    MATCH_SYNTHETIC = "match_synthetic"


@dataclass
class ImageFingerprint:
    """Perceptual hash fingerprint of an image."""
    url: str
    phash: int
    file_hash: str
    width: int = 0
    height: int = 0
    metadata: dict = field(default_factory=dict)

    def hamming_distance(self, other: 'ImageFingerprint') -> int:
        """Compute Hamming distance between two perceptual hashes."""
        xor = self.phash ^ other.phash
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    def similarity(self, other: 'ImageFingerprint') -> float:
        """Compute similarity score (0.0 to 1.0) based on perceptual hash."""
        distance = self.hamming_distance(other)
        return 1.0 - (distance / 64.0)


@dataclass
class ScanMatch:
    """A detected match between a found image and a reference."""
    source_url: str
    reference_path: str
    similarity: float
    is_synthetic: bool
    scan_timestamp: str
    synthetic_indicators: list = field(default_factory=list)


class PerceptualHasher:
    """Compute DCT-based perceptual hashes for images.

    Uses a simplified DCT approach that produces a 64-bit hash
    robust to resizing, compression, and minor edits.
    """

    HASH_SIZE = 8  # 8x8 = 64-bit hash

    @staticmethod
    def _dct_2d(matrix: list[list[float]]) -> list[list[float]]:
        """Compute 2D Discrete Cosine Transform."""
        n = len(matrix)
        result = [[0.0] * n for _ in range(n)]

        for u in range(n):
            for v in range(n):
                total = 0.0
                for x in range(n):
                    for y in range(n):
                        total += (
                            matrix[x][y]
                            * math.cos(math.pi * u * (2 * x + 1) / (2 * n))
                            * math.cos(math.pi * v * (2 * y + 1) / (2 * n))
                        )

                cu = (1 / math.sqrt(n)) if u == 0 else math.sqrt(2 / n)
                cv = (1 / math.sqrt(n)) if v == 0 else math.sqrt(2 / n)
                result[u][v] = cu * cv * total

        return result

    @classmethod
    def compute_hash(cls, pixels: list[list[float]]) -> int:
        """Compute perceptual hash from a grayscale pixel matrix.

        Args:
            pixels: 2D list of grayscale values (0.0 - 255.0),
                    should be resized to 32x32.

        Returns:
            64-bit perceptual hash as an integer.
        """
        dct = cls._dct_2d(pixels)

        # Take top-left 8x8 of DCT (low frequencies)
        low_freq = []
        for i in range(cls.HASH_SIZE):
            for j in range(cls.HASH_SIZE):
                low_freq.append(dct[i][j])

        # Compute median
        sorted_freq = sorted(low_freq)
        median = sorted_freq[len(sorted_freq) // 2]

        # Generate hash: 1 if above median, 0 if below
        phash = 0
        for val in low_freq:
            phash = (phash << 1) | (1 if val > median else 0)

        return phash

    @staticmethod
    def hash_to_hex(phash: int) -> str:
        """Convert a 64-bit hash to hex string."""
        return f"{phash:016x}"

    @staticmethod
    def hex_to_hash(hex_str: str) -> int:
        """Convert hex string back to hash integer."""
        return int(hex_str, 16)


class SyntheticDetector:
    """Analyze images for synthetic generation indicators.

    Current version uses metadata heuristics. Production version
    will use a fine-tuned ConvNeXt model.
    """

    # Known AI generation tool signatures in EXIF/metadata
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

    # Suspicious metadata patterns
    SUSPICIOUS_SOFTWARE = [
        "python",  # PIL/Pillow generation
        "pytorch",
        "tensorflow",
    ]

    @classmethod
    def analyze(cls, metadata: dict) -> tuple[bool, list[str]]:
        """Analyze image metadata for synthetic generation indicators.

        Args:
            metadata: Dict of image metadata (EXIF, XMP, etc.)

        Returns:
            Tuple of (is_synthetic, list of indicators found)
        """
        indicators = []

        # Check software field
        software = str(metadata.get("software", "")).lower()
        for sig in cls.SYNTHETIC_SIGNATURES:
            if sig in software:
                indicators.append(f"AI tool signature: {sig}")

        for sus in cls.SUSPICIOUS_SOFTWARE:
            if sus in software:
                indicators.append(f"Suspicious software: {sus}")

        # Check for AI generation parameters in metadata
        params = str(metadata.get("parameters", "")).lower()
        usercomment = str(metadata.get("usercomment", "")).lower()

        ai_params = ["cfg scale", "sampling steps", "sampler", "negative prompt",
                      "lora", "checkpoint", "vae", "clip skip"]
        for param in ai_params:
            if param in params or param in usercomment:
                indicators.append(f"AI generation parameter: {param}")

        # Check for C2PA provenance (content credentials)
        if "c2pa" in str(metadata).lower():
            indicators.append("C2PA content credential found")

        # Check for unusual resolution patterns (common in AI generation)
        width = metadata.get("width", 0)
        height = metadata.get("height", 0)
        if width and height:
            # AI tools often generate at exact multiples of 64
            if width % 64 == 0 and height % 64 == 0 and width >= 512:
                indicators.append(f"Resolution {width}x{height} matches AI generation pattern")

        is_synthetic = len(indicators) >= 2
        return is_synthetic, indicators


class TakedownGenerator:
    """Generate TAKE IT DOWN Act-compliant takedown requests."""

    TEMPLATE = """
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
  1. Remove the identified content immediately
  2. Prevent re-upload using the provided content hash
  3. Preserve evidence for potential law enforcement referral

============================================================
Generated by Athena — https://athena.ai
"""

    @classmethod
    def generate(cls, match: ScanMatch) -> str:
        """Generate a takedown request for a detected match."""
        platform = urlparse(match.source_url).netloc
        indicators = "\n".join(f"  - {ind}" for ind in match.synthetic_indicators)
        if not indicators:
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


class WebScanner:
    """Crawl web pages and extract images for analysis."""

    def __init__(self, reference_hashes: list[ImageFingerprint]):
        self.reference_hashes = reference_hashes
        self.matches: list[ScanMatch] = []
        self.scanned_urls: set[str] = set()
        self.similarity_threshold = 0.75

    def scan_url(self, url: str) -> list[ScanMatch]:
        """Scan a URL for matching images.

        In production, this would:
        1. Fetch the page HTML
        2. Extract all image URLs
        3. Download and hash each image
        4. Compare against reference hashes
        """
        print(f"  [*] Scanning: {url}")
        self.scanned_urls.add(url)
        # Production implementation would use requests + BeautifulSoup
        return []

    def scan_url_list(self, urls: list[str]) -> list[ScanMatch]:
        """Scan multiple URLs."""
        all_matches = []
        for url in urls:
            matches = self.scan_url(url)
            all_matches.extend(matches)
        return all_matches

    def report(self) -> dict:
        """Generate a scan report."""
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
    """Run a demonstration with synthetic test data."""
    print("=" * 60)
    print("  ATHENA IMAGE MONITOR — DEMO MODE")
    print("  Protecting women & girls from deepfake abuse")
    print("=" * 60)
    print()

    # Simulate reference hashes
    ref = ImageFingerprint(
        url="protected_user_ref_01.jpg",
        phash=0xA3B1C2D4E5F60718,
        file_hash="abc123",
    )

    # Simulate scan results
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
    for result in demo_results:
        if result.similarity >= 0.75 and result.is_synthetic:
            status = "MATCH [SYNTHETIC]"
            icon = "\U0001f6a8"
            flagged.append(result)
            print(f"  {icon} {status}")
            print(f"    URL:        {result.source_url}")
            print(f"    Similarity: {result.similarity:.0%}")
            print(f"    Matched:    {result.reference_path}")
        else:
            print(f"  \u2713 Clear")
            print(f"    URL:        {result.source_url}")
            print(f"    Similarity: {result.similarity:.0%}")
        print()

    if flagged:
        print(f"[*] Generating {len(flagged)} TAKE IT DOWN Act takedown request(s)...")
        print()
        for match in flagged:
            request = TakedownGenerator.generate(match)
            print(request)


def main():
    parser = argparse.ArgumentParser(
        description="Athena Image Monitor — Detect non-consensual AI imagery"
    )
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--refs", type=str, help="Path to reference photos directory")
    parser.add_argument("--url", type=str, help="Single URL to scan")
    parser.add_argument("--urls", type=str, help="File containing URLs to scan (one per line)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold (0-1)")
    parser.add_argument("--output", type=str, help="Output report as JSON")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.refs:
        parser.error("--refs is required (or use --demo)")

    ref_path = Path(args.refs)
    if not ref_path.exists():
        print(f"Error: Reference path {ref_path} does not exist")
        sys.exit(1)

    # Load reference images
    print(f"[*] Loading reference images from {ref_path}")
    references = []
    for img_file in ref_path.glob("*"):
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            fp = ImageFingerprint(
                url=str(img_file),
                phash=int(hashlib.md5(img_file.read_bytes()).hexdigest()[:16], 16),
                file_hash=hashlib.sha256(img_file.read_bytes()).hexdigest(),
            )
            references.append(fp)
            print(f"  Enrolled: {img_file.name}")

    print(f"[*] {len(references)} reference image(s) enrolled")

    # Scan URLs
    scanner = WebScanner(references)
    scanner.similarity_threshold = args.threshold

    urls = []
    if args.url:
        urls.append(args.url)
    if args.urls:
        with open(args.urls) as f:
            urls.extend(line.strip() for line in f if line.strip())

    if not urls:
        parser.error("Provide --url or --urls to scan")

    print(f"[*] Scanning {len(urls)} URL(s)...")
    scanner.scan_url_list(urls)

    # Output report
    report = scanner.report()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[*] Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
