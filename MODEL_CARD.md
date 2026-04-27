# Model Card · Athena Synthetic-Image Classifier

A short model card in the style of *Mitchell et al., FAccT 2019*.

## Model details

- **Developed by:** Maisy Mylod / Athena
- **Model type:** Binary image classifier (real vs AI-generated)
- **Architecture:** EfficientNet-B0 (ImageNet pretrained) + custom 2-layer
  classification head; trained with two-phase fine-tuning. See
  `ml/model.py` and `ml/train.py`.
- **Inputs:** RGB images, resized to 224×224, normalized with ImageNet
  statistics.
- **Output:** A single logit, sigmoid-converted to a probability that the
  image is AI-generated. Returned with a small list of supporting
  indicators.
- **License:** Code under the repo's license. Trained weights are not
  redistributed under a permissive license; see "Training data" below.

## Intended use

**In scope:**

1. Helping a person decide whether an image they have received or found is
   likely to be AI-generated.
2. As a verification step inside the Athena consumer flow: once a perceptual
   hash collision is found against an enrollee's reference photo, the
   classifier confirms the suspect image is synthetic before drafting a
   takedown.
3. Educational and journalistic use ("is this image real?").

**Out of scope:**

1. Standalone evidence in a legal proceeding. The classifier is not
   forensically certified and should not be used in court without expert
   review.
2. Identifying *who* is in an image. The model is a real-vs-synthetic
   classifier, not a face-recognition or face-matching system.
3. Detecting traditional (non-AI) image manipulation such as Photoshop
   composites, splicing, or analog forgeries.
4. Detecting deepfakes in video. Video-level cues (temporal artifacts,
   eye-blink patterns, lip-sync inconsistencies) are not modeled.

## Training data

See [`EVAL.md`](EVAL.md) for the dataset breakdown. Briefly: a public
real-vs-synthetic image dataset, stratified by label into train / val /
test, with augmentations designed to simulate real-world recompression
and resizing.

We do not train on any non-consensual intimate imagery. The training
distribution skews safe-for-work; performance on extremely high-frequency
or low-resolution unsafe-for-work content has not been characterized.

## Performance

Headline numbers and robustness tables in [`EVAL.md`](EVAL.md). Calibration
is not yet measured. The confidence values returned by the API are sigmoid
probabilities, not calibrated likelihoods.

## Ethical considerations

Athena's broader product is a face-monitoring service for people who
believe they are at risk of non-consensual deepfake imagery. That product
has the same capability surface as a stalker tool. We design and gate
deployment of the consumer product around four mitigations:

1. **Identity-verified enrollment.** Enrollment requires a government ID
   and a selfie liveness check whose face must match the enrolled
   reference. There is no path to enroll someone else's face.
2. **Hash-only storage.** Reference photos are converted into a one-way
   perceptual hash and the originals are discarded. We never store raw
   face embeddings or biometric vectors that could be reused as a face DB.
3. **Match notifications go only to the enrollee.** No admin console can
   look up "who matches this hash"; no third-party feed is exposed.
4. **Tamper-evident audit log.** Every read against the hash store is
   logged.

The classifier itself does not surface any face identity, only a
synthetic-vs-real signal, and is therefore not subject to the same
abuse risks as the broader product. If you are integrating it into
a higher-risk system, the burden of those mitigations is on you.

## Known failure modes

- **Cross-generator drift.** Detectors trained on one generation tool
  (e.g. Stable Diffusion) often degrade on samples from other generators.
  See cross-generator numbers in `EVAL.md`.
- **Aggressive recompression.** Quality < 30 JPEGs and very low resolutions
  (< 128 px on the long edge) are out of distribution for our augmentations.
- **Watermark bias.** Some training corpora include watermarks ("SDXL",
  "Generated with…") that the model may anchor on. We have not yet run a
  watermark-removal ablation.
- **Adversarial inputs.** No defense against pixel-level adversarial
  attacks has been evaluated.

## Reporting issues

Open an issue at https://github.com/maisymylod/athena-ai/issues. For
abuse reports involving the consumer product, the in-product abuse
reporting channel takes priority.
