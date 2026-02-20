# Athena

**Protecting women & girls from deepfake abuse.**

Athena detects AI-generated non-consensual intimate imagery, monitors the web for deepfakes of protected individuals, and automates [TAKE IT DOWN Act](https://www.congress.gov/bill/119th-congress/senate-bill/146) takedown requests.

96% of deepfakes are non-consensual pornography. 99% of the victims are women. There are zero consumer tools to fight back. Athena changes that.

---

## What's in this repo

```
athena-ai/
├── index.html              # Landing page (athena.ai)
├── tools/
│   └── monitor.py          # Image monitoring & scraping pipeline
├── requirements.txt
└── README.md
```

### Landing Page

A static site for athena.ai with waitlist signup. Open `index.html` in any browser.

### Image Monitor (`tools/monitor.py`)

A proof-of-concept web scanning pipeline that:

- **Crawls** web pages and extracts image URLs
- **Computes perceptual hashes** (pHash) of found images
- **Compares** against a set of protected reference images to find potential matches
- **Analyzes** images for synthetic generation indicators (metadata heuristics, with a stub for the trained ML model)
- **Generates TAKE IT DOWN Act-compliant takedown requests** for matched content

#### Quick start

```bash
pip install -r requirements.txt

# Run the demo (no external requests, uses synthetic test data)
python tools/monitor.py --demo

# Scan a URL against your reference photos
python tools/monitor.py --refs ./my_photos/ --url https://example.com

# Scan a list of URLs
python tools/monitor.py --refs ./my_photos/ --urls watchlist.txt
```

#### Demo output

```
============================================================
  ATHENA IMAGE MONITOR — DEMO MODE
  Protecting women & girls from deepfake abuse
============================================================

[*] Scan Results:
------------------------------------------------------------
  🚨 MATCH [SYNTHETIC]
    URL:        https://example-site.com/images/img_0847.jpg
    Similarity: 94%
    Matched:    protected_user_ref_01.jpg

  ✓ Clear
    URL:        https://example-site.com/images/img_1293.jpg
    Similarity: 12%

  🚨 MATCH [SYNTHETIC]
    URL:        https://example-site.com/images/img_2041.jpg
    Similarity: 87%
    Matched:    protected_user_ref_01.jpg

[*] Generating 2 TAKE IT DOWN Act takedown request(s)...
```

---

## How it works

1. **Identity enrollment** — Upload reference photos. Athena computes perceptual hashes (DCT-based pHash) that are robust to resizing, compression, and minor edits.

2. **Web monitoring** — The scraping pipeline crawls target sites and image boards, extracting images and comparing their perceptual hashes against enrolled identities.

3. **Deepfake detection** — Images are analyzed for synthetic generation artifacts. The current version uses metadata heuristics; the production version will use a fine-tuned ConvNeXt model trained against output from top nudify tools.

4. **Automated takedowns** — When a match is confirmed, Athena generates a TAKE IT DOWN Act-compliant takedown notice and files it with the hosting platform.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Detection model (planned) | PyTorch, ConvNeXt/EfficientNet, DCT spectral analysis |
| On-device inference | ONNX Runtime, Core ML, TensorFlow Lite |
| Image matching | Perceptual hashing (pHash), CLIP embeddings |
| Web scraping | Scrapy, BeautifulSoup, Requests |
| Backend | Python (FastAPI), PostgreSQL, Redis |
| Mobile | React Native |
| Infrastructure | AWS (EC2, S3, Lambda) |

## Roadmap

- [x] Landing page with waitlist
- [x] Perceptual hashing engine
- [x] Web scraping/crawling pipeline
- [x] Takedown request generator
- [x] Synthetic image analysis (heuristic)
- [ ] Trained deepfake detection model (ConvNeXt fine-tune)
- [ ] Frequency-domain analysis (DCT spectral fingerprints)
- [ ] CLIP-based semantic similarity matching
- [ ] iOS / Android app
- [ ] Platform reporting API integrations
- [ ] Real-time monitoring dashboard

## The problem

| Stat | Source |
|------|--------|
| 99% of deepfake porn targets women | Sensity AI |
| 440,000 child deepfake reports to NCMEC in H1 2025 | NCMEC |
| 21 million monthly visits to nudify websites | ISD Global |
| 1 in 10 minors say classmates use AI to generate nudes | Thorn |
| $0 consumer tools exist to combat this | — |

## Built by

**Maisy Mylod** — B.S. Pure Mathematics, University of Michigan. Former Data Analyst at CLEAR (biometric identity, 20M+ users). Software Engineer at Goldsmith & Co.

[LinkedIn](https://linkedin.com/in/maisymylod) · [GitHub](https://github.com/maisymylod)

---

*Athena is named after the Greek goddess of wisdom, courage, and strategic defense.*
