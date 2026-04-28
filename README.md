# Athena

**Detect AI-generated non-consensual intimate imagery, prove it's synthetic, and get it removed under the [TAKE IT DOWN Act](https://www.congress.gov/bill/119th-congress/senate-bill/146).**

[![CI](https://github.com/maisymylod/athena-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/maisymylod/athena-ai/actions/workflows/ci.yml)

96% of deepfakes are non-consensual pornography. 99% of victims are women. Zero consumer tools exist to fight back. Athena is what changes that.

- **Live demo:** [**mymaisy-athena-detect.hf.space**](https://mymaisy-athena-detect.hf.space/) — drag in any image, get a verdict in ~150ms
- **Detector evaluation:** [`EVAL.md`](EVAL.md)
- **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md)

## What's shipped today

A real-vs-AI image classifier (EfficientNet-B0, two-phase fine-tune) you can hit at `POST /api/detect`. Drop an image in; get back a synthetic-vs-real verdict, a confidence score, and the indicators that drove the call.

```bash
pip install -r requirements.txt
python -m flask --app server.app:create_app run --port 5000
# in another shell:
curl -F image=@your_test.jpg http://localhost:5000/api/detect
```

The drag-and-drop UI in `index.html` is wired to the same endpoint. Six "Try with these" example thumbnails are bundled in `static/examples/` for one-click testing.

## What's on the roadmap

The longer-term consumer product is a face-monitoring service for people at risk of non-consensual deepfakes:

1. **Identity-verified enrollment.** Government ID + a selfie liveness check whose face has to match the enrolled reference. You can only enroll your own face.
2. **Hash-only storage.** Reference photos are converted into a one-way perceptual hash; originals are discarded.
3. **Match notifications go only to the enrollee.** No admin lookup, no third-party feed, full audit log.
4. **Auto-drafted TAKE IT DOWN Act takedowns** when the classifier verifies a hash match is also synthetic.

The capability surface for that product overlaps with stalker tooling, so we are designing those four mitigations into the first deployable version, not as bolt-ons. See `MODEL_CARD.md` for the long version.

## Repo layout

```
athena-ai/
├── index.html              # Landing page + drag-drop demo UI
├── server/app.py           # Flask API: /api/detect, /api/health
├── ml/                     # PyTorch training & inference pipeline
│   ├── model.py            # EfficientNet-B0 + classification head
│   ├── train.py            # 2-phase trainer (frozen backbone → fine-tune)
│   ├── dataset.py          # Stratified split + augmentations (incl. JPEG)
│   ├── inference.py        # Loads a checkpoint, runs predict()
│   └── evaluate.py         # Accuracy / precision / recall / F1 / ROC-AUC
├── tools/monitor.py        # Perceptual-hash matching + takedown drafting
├── scripts/run_training.py # Wrapper around ml.train
├── static/                 # Frontend assets + bundled demo examples
├── tests/                  # pytest suite (CI-gated)
├── Dockerfile, fly.toml    # Demo deployment
└── EVAL.md, MODEL_CARD.md  # Detector eval + model card
```

## Train your own classifier

```bash
# 1. Get a dataset
python ml/download_dataset.py --list
mkdir -p data/raw/real data/raw/synthetic
# (drop images into the appropriate subdirectories)

# 2. Train
python scripts/run_training.py --data-dir data/raw --output-dir checkpoints

# 3. Run the API against the new weights
ATHENA_CHECKPOINT=checkpoints/best_model.pt \
    python -m flask --app server.app:create_app run
```

Two-phase training runs by default: phase 1 freezes the backbone and trains only the classifier head; phase 2 unfreezes at 1/10 the learning rate. Configurable in `ml/config.py:TrainConfig`.

## The problem

| Stat | Source |
|------|--------|
| 99% of deepfake porn targets women | Sensity AI |
| 440,000 child deepfake reports to NCMEC in H1 2025 | NCMEC |
| 21M monthly visits to nudify websites | ISD Global |
| 1 in 10 minors say classmates use AI to generate nudes of other kids | Thorn |
| $0 consumer tools exist | — |

## Built by

**Maisy Mylod** — B.S. Pure Mathematics, University of Michigan. Former Data Analyst at CLEAR (biometric identity, 20M+ users). Software Engineer at Goldsmith & Co.

[LinkedIn](https://linkedin.com/in/maisymylod) · [GitHub](https://github.com/maisymylod)

---

*Athena is named after the Greek goddess of wisdom, courage, and strategic defense.*
