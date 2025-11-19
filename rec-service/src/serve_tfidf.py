from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import scipy.sparse as sp
import os
import re
import json
from .auth import router as auth_router
from .db import Base, engine

app = FastAPI(title="Scent2Me Recommendation API")

# ------------------------------
# CORS
# ------------------------------
ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "http://localhost:3000,"
    "https://scent2me-h8364io4x-denjears-projects.vercel.app,"
    "https://scent2me.vercel.app"

).split(",")

origins = [o.strip().rstrip("/") for o in ALLOW_ORIGINS if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------
# Load artifacts
# ------------------------------
ART_DIR = os.getenv("ART_DIR", "./artifacts")
meta_path = os.path.join(ART_DIR, "meta.csv")
tfidf_path = os.path.join(ART_DIR, "tfidf.pkl")
matrix_path = os.path.join(ART_DIR, "X_tfidf.npz")

print(f"🔹 Loading artifacts from {ART_DIR} ...")
meta = pd.read_csv(meta_path)
tfidf = load(tfidf_path)
X_tfidf = sp.load_npz(matrix_path)

# Ensure essential columns exist
ESSENTIAL = ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num", "gender", "bag_text"]
for col in ESSENTIAL:
    if col not in meta.columns:
        meta[col] = np.nan

# Normalize some textual fields for matching
def _norm(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

meta["_gender_lc"] = meta["gender"].astype(str).map(_norm)
meta["_bag_lc"] = meta["bag_text"].astype(str).map(_norm)

print(f"✅ Loaded {len(meta)} perfumes")

# ------------------------------
# Schemas
# ------------------------------
class PreferenceRequest(BaseModel):
    # legacy fields (tetap diterima)
    gender: Optional[str] = None                         # "male|female|unisex"
    family: Optional[Union[List[str], str]] = None       # legacy: single or list
    context: Optional[str] = None                        # "morning|evening|..."
    longevity: Optional[str] = None                      # "strong|moderate|long"
    top_k: int = 12

    # new fields (opsional)
    families: Optional[List[str]] = None                 # new: multi-select
    time_of_day: Optional[str] = None                    # new
    occasion: Optional[str] = None                       # new: "daily|office|date|party|formal"
    performance: Optional[str] = None                    # new: "light|moderate|strong"
    notes: Optional[List[str]] = None                    # new: ["rose","vanilla"]
    min_price: Optional[float] = None                    # new
    max_price: Optional[float] = None                    # new


# ------------------------------
# Health
# ------------------------------
@app.get("/health")
def health_check():
    return {"ok": True, "n_items": int(len(meta))}


# ------------------------------
# GET /search  (search by name and/or brand)
# ------------------------------
@app.get("/search")
def search(name: Optional[str] = Query(None), brand: Optional[str] = Query(None), top_k: int = 12):
    if not name and not brand:
        return {"error": "Provide at least 'name' or 'brand' query parameter"}

    mask = np.ones(len(meta), dtype=bool)

    if name:
        name_lc = _norm(name)
        mask = mask & meta["name_display"].astype(str).map(_norm).str.contains(name_lc, na=False).values

    if brand:
        brand_lc = _norm(brand)
        mask = mask & meta["brand_display"].astype(str).map(_norm).str.contains(brand_lc, na=False).values

    if not mask.any():
        return {"count": 0, "results": []}

    order = np.where(mask)[0][:top_k]

    cols = ["id"] if "id" in meta.columns else []
    cols += ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num"]
    result = meta.iloc[order][cols].copy()

    return {"count": int(len(result)), "results": result.to_dict(orient="records")}


# ------------------------------
# GET /trending  (top-rated perfumes)
# ------------------------------
@app.get("/trending")
def trending(top_k: int = 12):
    # Sort by rating_num descending, drop NaN
    sorted_meta = meta.dropna(subset=["rating_num"]).sort_values("rating_num", ascending=False)
    order = sorted_meta.index[:top_k]

    cols = ["id"] if "id" in meta.columns else []
    cols += ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num"]
    result = meta.iloc[order][cols].copy()

    return {"count": int(len(result)), "results": result.to_dict(orient="records")}


# ------------------------------
# GET /random  (random perfume)
# ------------------------------
@app.get("/random")
def random_perfume():
    idx = np.random.choice(len(meta))
    cols = ["id"] if "id" in meta.columns else []
    cols += ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num"]
    result = meta.iloc[[idx]][cols].copy()

    return {"count": 1, "results": result.to_dict(orient="records")}


# ------------------------------
# GET /brands  (unique brands for dropdown)
# ------------------------------
@app.get("/brands")
def get_brands():
    brands = meta["brand_display"].dropna().unique().tolist()
    return {"brands": sorted(brands)}


# ------------------------------
# GET /recommend  (by reference perfume)
# ------------------------------
@app.get("/recommend")
def recommend(perfume_name: str = Query(...), top_k: int = 5):
    name_lc = _norm(perfume_name)

    # cari case-insensitive
    idx_candidates = meta.index[meta["name_display"].astype(str).map(_norm) == name_lc].tolist()
    if not idx_candidates:
        return {"error": f"'{perfume_name}' not found"}
    idx = idx_candidates[0]

    sims = cosine_similarity(X_tfidf[idx], X_tfidf).flatten()

    # ambil top_k selain diri sendiri
    order = np.argsort(sims)[::-1]
    order = order[order != idx][:top_k]

    cols = ["id"] if "id" in meta.columns else []
    cols += ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num"]
    result = meta.iloc[order][cols].copy()
    result["similarity"] = sims[order]
    return result.to_dict(orient="records")


# ------------------------------
# POST /recommend/preference  (by user preference)
# ------------------------------
@app.post("/recommend/preference")
def recommend_by_preference(req: PreferenceRequest):
    try:
        # --- 1) Normalisasi input & backward compatibility ---
        # families: prefer new field, fallback ke legacy family
        fams: List[str] = []
        if req.families:
            fams = [ _norm(x) for x in req.families if x ]
        elif req.family is not None:
            if isinstance(req.family, list):
                fams = [ _norm(x) for x in req.family if x ]
            elif isinstance(req.family, str):
                fams = [ _norm(req.family) ] if req.family else []

        gender = _norm(req.gender or "unisex")
        time_of_day = _norm(req.time_of_day or req.context or "")
        performance = _norm(req.performance or req.longevity or "moderate")
        occasion = _norm(req.occasion or "")
        notes = [ _norm(x) for x in (req.notes or []) if x ]

        top_k = max(1, int(req.top_k or 12))
        min_price = req.min_price
        max_price = req.max_price

        # --- 2) Prefilter ringan (subset) ---
        # Start with all candidates
        mask = np.ones(len(meta), dtype=bool)

        # Gender: pilih gender sama + unisex
        if gender in {"male", "female", "unisex"}:
            mask = mask & (
                (meta["_gender_lc"].values == gender) |
                (meta["_gender_lc"].values == "unisex")
            )

        # Families: cari kata di bag_text (regex word boundary sederhana)
        if fams:
            pat = r"|".join([rf"\b{re.escape(f)}\b" for f in fams])
            mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values

        # Time of day → token preferensi ringan
        if time_of_day:
            # mapping ringan; bisa kamu sesuaikan dengan data nyata
            tod_map = {
                "morning": ["fresh", "citrus", "green", "aquatic", "aromatic"],
                "afternoon": ["floral", "fresh", "citrus"],
                "evening": ["woody", "spicy", "oriental", "amber"],
                "night": ["oriental", "gourmand", "oud", "leather"],
            }
            toks = tod_map.get(time_of_day, [])
            if toks:
                pat = r"|".join([rf"\b{re.escape(t)}\b" for t in toks])
                mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values

        # Occasion (opsional contoh)
        if occasion:
            occ_map = {
                "daily": ["clean", "fresh", "citrus", "light"],
                "office": ["clean", "subtle", "powdery", "musky"],
                "date": ["vanilla", "amber", "rose", "sweet"],
                "party": ["intense", "spicy", "loud", "oud"],
                "formal": ["amber", "woody", "leather"],
            }
            toks = occ_map.get(occasion, [])
            if toks:
                pat = r"|".join([rf"\b{re.escape(t)}\b" for t in toks])
                mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values

        # Performance: light / moderate / strong
        if performance:
            if performance == "light":
                pat = r"fresh|cologne|\bedt\b|clean"
                mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values
            elif performance == "strong":
                pat = r"parfum|amber|oud|intense|smoky"
                mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values
            # moderate → no additional filter (default)

        # Notes (keywords)
        if notes:
            pat = r"|".join([rf"\b{re.escape(n)}\b" for n in notes])
            mask = mask & meta["_bag_lc"].str.contains(pat, regex=True, na=False).values

        # Price range
        if min_price is not None:
            mask = mask & (meta["price_num"].fillna(0).values >= float(min_price))
        if max_price is not None:
            mask = mask & (meta["price_num"].fillna(0).values <= float(max_price))

        # Jika filter terlalu ketat, longgarkan sedikit (fallback)
        if not mask.any():
            mask = np.ones(len(meta), dtype=bool)

        # --- 3) Scoring TF-IDF (safe masking) ---
        # Build a simple query text dari preferensi
        query_parts = [gender]
        if fams:
            query_parts.extend(fams)
        if time_of_day:
            query_parts.append(time_of_day)
        if occasion:
            query_parts.append(occasion)
        if performance:
            query_parts.append(performance)
        if notes:
            query_parts.extend(notes)

        query_text = " ".join([p for p in query_parts if p])
        if not query_text.strip():
            query_text = "unisex moderate"  # default yang netral

        print(f"🔍 preference query: {query_text}")

        query_vec = tfidf.transform([query_text])
        sims = cosine_similarity(query_vec, X_tfidf).flatten()

        # Terapkan mask: kandidat yang tidak lolos filter diberi -inf
        sims_masked = sims.copy()
        sims_masked[~mask] = -np.inf

        order = np.argsort(sims_masked)[::-1]
        order = order[np.isfinite(sims_masked[order])][:top_k]  # buang -inf

        if order.size == 0:
            return {"query": query_text, "count": 0, "results": []}

        # --- 4) Susun respons ---
        cols = ["id"] if "id" in meta.columns else []
        cols += ["name_display", "brand_display", "image_url", "buy_url", "price_num", "rating_num", "bag_text"]

        result = meta.iloc[order][cols].copy()
        result["similarity"] = sims[order]

        print(f"✅ Returned {len(result)} recommendations")

        return {
            "query": {
                "text": query_text,
                "gender": gender,
                "families": fams,
                "time_of_day": time_of_day,
                "occasion": occasion,
                "performance": performance,
                "notes": notes,
                "min_price": min_price,
                "max_price": max_price,
            },
            "count": int(len(result)),
            "results": result.to_dict(orient="records"),
        }

    except Exception as e:
        print("❌ Error in /recommend/preference:", e)
        return {"error": str(e)}

# ------------------------------
# WISHLIST ENDPOINTS
# ------------------------------

WISHLIST_PATH = os.path.join(ART_DIR, "wishlist.json")

# pastikan file wishlist ada
if not os.path.exists(WISHLIST_PATH):
    # create as a per-user mapping by default (avoids accidental global/shared list)
    with open(WISHLIST_PATH, "w") as f:
        json.dump({}, f)


@app.get("/wishlist")
def get_wishlist():
    """Ambil semua parfum yang tersimpan di wishlist"""
    try:
        with open(WISHLIST_PATH, "r") as f:
            wishlist = json.load(f)
        return {"wishlist": wishlist}
    except Exception as e:
        return {"error": f"Failed to read wishlist: {e}"}


@app.post("/wishlist/add")
async def add_to_wishlist(request: Request):
    """Tambah satu atau beberapa parfum ke wishlist"""
    try:
        payload = await request.json()

        # payload can be:
        # - a list of items (legacy)
        # - a dict { email: 'user@email', items: [...] }
        # - a single item dict
        email = None
        items = []
        if isinstance(payload, dict) and "items" in payload:
            email = payload.get("email")
            items = payload.get("items") or []
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            # single item
            items = [payload]

        with open(WISHLIST_PATH, "r") as f:
            wishlist = json.load(f)

        # If wishlist on disk is a mapping (per-user), store under the user's key
        if isinstance(wishlist, dict):
            # When storage is per-user mapping, require the client to provide an email
            if not email:
                # Do not silently write to a shared 'global' bucket when mapping exists
                raise HTTPException(status_code=400, detail="Missing email in payload for per-user wishlist storage")

            key = email
            if key not in wishlist or not isinstance(wishlist[key], list):
                wishlist[key] = []

            existing_names = {item.get("name_display") for item in wishlist[key] if isinstance(item, dict) and "name_display" in item}
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("name_display") not in existing_names:
                    wishlist[key].append(item)

        else:
            # legacy: wishlist is a list
            if not isinstance(wishlist, list):
                wishlist = []
            existing_names = {item.get("name_display") for item in wishlist if isinstance(item, dict) and "name_display" in item}
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("name_display") not in existing_names:
                    wishlist.append(item)

        with open(WISHLIST_PATH, "w") as f:
            json.dump(wishlist, f, indent=2)

        # compute total count depending on structure
        total = len(wishlist) if isinstance(wishlist, list) else sum(len(v) for v in wishlist.values())
        return {"message": "Added to wishlist", "count": int(total)}

    except Exception as e:
        return {"error": f"Failed to add wishlist: {e}"}


@app.post("/wishlist/clear")
async def clear_wishlist(request: Request):
    """Hapus wishlist. If storage is a mapping, require { "email": "..." } in the body
    and only clear that user's list. If storage is a legacy list, fully clear it.
    """
    try:
        # try to read optional payload
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        email = None
        if isinstance(payload, dict):
            email = payload.get("email")

        with open(WISHLIST_PATH, "r") as f:
            wishlist = json.load(f)

        if isinstance(wishlist, dict):
            # per-user mapping: require email to clear only that user's list
            if not email:
                raise HTTPException(status_code=400, detail="Missing email in request body for clearing user wishlist")
            wishlist[email] = []
        else:
            # legacy list: clear whole file
            wishlist = []

        with open(WISHLIST_PATH, "w") as f:
            json.dump(wishlist, f, indent=2)

        return {"message": "Wishlist cleared"}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"Failed to clear wishlist: {e}"}

Base.metadata.create_all(bind=engine)

# Include auth router
app.include_router(auth_router, prefix="/auth")

@app.get("/")
async def root():
    return {"message": "API is running"}
