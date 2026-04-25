# TrustVerify ML Service

This service contains the TrustVerify machine learning inference API used by the backend.

## Models

- Text: text authenticity detection pipeline.
- Voice: voice authenticity detection pipeline.
- Image: image authenticity detection pipeline or integration point.

Training datasets and generated model artifacts are removed from the GitHub version of this project. Recreate or download approved data and model files outside Git history when needed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Inference

Start the service:

```bash
uvicorn app.main:app --reload
```

Use the API routes under `app/api/routes` for text and voice inference. Model loaders expect runtime model artifacts to be supplied locally or through your deployment process, not committed to Git.
