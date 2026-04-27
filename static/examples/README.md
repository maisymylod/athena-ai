# Example images for the demo

These six images power the "Try with these" affordance on the landing page.
They are held-out test samples from
[Madushan996/real_fake_images](https://huggingface.co/datasets/Madushan996/real_fake_images),
the same dataset whose training portion was used to fine-tune the classifier.

- `real_1.jpg`, `real_2.jpg`, `real_3.jpg` — real photos
- `ai_1.jpg`, `ai_2.jpg`, `ai_3.jpg` — AI-generated images, one each from
  three different generators (Flux, Stable Diffusion 1.5, plus one more)

To regenerate them (e.g. to pick a different held-out sample, or after
moving to a different dataset), run:

```bash
python scripts/prepare_dataset.py
```

This script also rebuilds `data/raw/{real,synthetic}/` for training, so
the test images are guaranteed to be disjoint from what the model trained
on.
