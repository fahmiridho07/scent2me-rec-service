# scent2me-rec-service

Content-based **perfume recommendation service** for the Scent2Me app. It ranks fragrances by similarity using a **TF-IDF** model (scikit-learn) and serves recommendations through a **FastAPI** REST API with JWT authentication and a PostgreSQL backend.

## Features

- **Content-based recommender** using TF-IDF vectorization + cosine similarity over fragrance attributes.
- **FastAPI** service exposing recommendation endpoints (`src/serve_tfidf.py`).
- **JWT authentication** (`python-jose`) with password hashing via `passlib[bcrypt]`.
- **Database layer** with SQLModel / SQLAlchemy 2.0 on PostgreSQL (`psycopg2`).
- Reusable model artifacts stored under `rec-service/artifacts/`.

## Tech stack

Python · scikit-learn · FastAPI · Uvicorn · pandas / NumPy / SciPy · SQLModel · SQLAlchemy · PostgreSQL · JWT

## Project structure

```
rec-service/
├── main.py              # FastAPI entrypoint (imports app from src/serve_tfidf.py)
├── requirements.txt
├── run.ps1              # Local run script (Windows)
├── artifacts/           # Saved TF-IDF model / vectorizer artifacts
└── src/
    ├── serve_tfidf.py   # FastAPI app + recommendation endpoints
    ├── auth.py          # JWT auth & password hashing
    ├── db.py            # Database session/config
    └── models.py        # Data models (SQLModel)
```

## Getting started

```bash
cd rec-service
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables (e.g. DATABASE_URL, JWT secret) in a .env file
uvicorn main:app --reload
```

Once running, open the interactive API docs at `http://localhost:8000/docs`.

## Notes

This service is the recommendation backend of the larger **Scent2Me** project.
