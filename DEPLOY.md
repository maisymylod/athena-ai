# Deploying the Athena demo to Fly.io

The repo ships a `Dockerfile` and `fly.toml` that deploy the Flask
classifier to a small shared-CPU machine on Fly.io's free tier. This
guide is the step-by-step.

## One-time prerequisites

1. Install the Fly CLI:
   ```bash
   brew install flyctl
   ```
2. Sign up + log in:
   ```bash
   fly auth signup       # or `fly auth login` if you already have an account
   ```
   Fly.io's free tier requires a credit card on file (for fraud
   prevention only). You will not be charged for the spec used in
   `fly.toml` (one shared-cpu-1x machine, 1 GB RAM, auto-stop when idle).

## Train the model first

The Dockerfile bakes `checkpoints/best_model.pt` into the image, so train
locally before deploying:

```bash
python scripts/prepare_dataset.py             # downloads dataset + writes data/raw/
python scripts/run_training.py \
    --data-dir data/raw --output-dir checkpoints \
    --epochs 12 --freeze-epochs 3 --batch-size 32 --num-workers 0
```

When training finishes, `checkpoints/best_model.pt` should be ~30 MB.

## Launch the app (first deploy only)

```bash
fly launch --no-deploy --copy-config --name athena-demo
```

`--copy-config` makes `fly launch` reuse the in-repo `fly.toml`; if the
app name `athena-demo` is taken, pick another (`fly launch` will prompt).

If `fly launch` re-runs and changes `fly.toml`, revert any unwanted
edits. The in-repo `fly.toml` is opinionated about the healthcheck path
and machine size.

## Deploy

```bash
fly deploy
```

Fly will build the Docker image, push it, and start one machine. First
build downloads ~700 MB of CPU torch wheels and takes 5–10 minutes; the
push of the resulting image is much smaller because of layer caching.

When deploy succeeds, your URL is `https://<app-name>.fly.dev`.

Sanity-check it:

```bash
curl https://<app-name>.fly.dev/api/health
# → {"checkpoint_path":"/app/checkpoints/best_model.pt","model_available":true,"status":"ok"}

curl -F image=@static/examples/real_1.jpg https://<app-name>.fly.dev/api/detect
# → {"is_synthetic":false,"confidence":...,"label":"Real","indicators":[...]}
```

## Re-deploy after changes

```bash
fly deploy
```

Caches Docker layers, so subsequent deploys are fast (1–2 minutes) unless
you change `requirements.txt` or `Dockerfile`.

## Updating just the checkpoint

If you re-train and only the checkpoint changes:

```bash
# Force-rebuild without no-cache; Docker will only refresh the
# `COPY checkpoints/` layer and everything that follows it.
fly deploy
```

## Cost monitoring

```bash
fly status
fly machines list
```

The configured machine auto-stops when idle (no requests for ~5 min) and
auto-starts on the next request. Cold start is ~10 s. While stopped, you
pay nothing.

## Custom domain (optional, ~$12/yr)

Skip until you've decided the brand name. To wire later:

1. Buy `tryathena.com` (or whichever) at any registrar.
2. `fly certs add tryathena.com`
3. Add the CNAME record Fly prints to your DNS.
4. `fly certs check tryathena.com` confirms issuance.

Until then, the `*.fly.dev` URL is fine for the YC application.
