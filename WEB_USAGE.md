# Web upload inference

This adds a small Flask + Jinja2 web app for uploading one image and returning anomaly
detection outputs from the SSPTT model.

## Files

- `web_app/app.py`: Flask backend and model inference
- `web_app/templates/index.html`: Jinja2 upload UI
- `web_app/static/app.js`: browser-side upload logic
- `web_app/static/styles.css`: page styling
- `requirements-web.txt`: extra packages for the web service

## Checkpoint

The backend needs a trained `.pth` checkpoint before it can predict. By default
it looks for:

```text
checkpoints/carpet_ep300.pth
```

You can point it to another checkpoint with:

```powershell
$env:SSPTT_CHECKPOINT="C:\path\to\your_checkpoint.pth"
```

The threshold defaults to `0.5`. You can override it with:

```powershell
$env:SSPTT_THRESHOLD="0.45"
```

## Install and run

```powershell
pip install -r requirements-web.txt
python web_app\app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## API

Health check:

```text
GET /api/health
```

Upload image:

```text
POST /api/predict
form-data field: file
```

Response includes:

- `score`: max anomaly score
- `label`: `normal` or `abnormal`
- `images.original`: base64 PNG
- `images.heatmap`: base64 PNG
- `images.overlay`: base64 PNG
- `images.mask`: base64 PNG

The first prediction can take a while because DINOv2 is loaded lazily through
`torch.hub`.
