from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import os

from .db import get_db
from .models import User, WishlistItem

router = APIRouter()

# ------------------------------
# Password hashing
# ------------------------------
pwd_context = CryptContext(
    schemes=["sha256_crypt"],
    deprecated="auto",
)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


# ------------------------------
# Pydantic models
# ------------------------------
class UserRegister(BaseModel):
    email: str
    password: str
    username: str


class UserLogin(BaseModel):
    email: str
    password: str


class Product(BaseModel):
    id: Optional[str] = None
    image_url: Optional[str] = None
    name_display: str
    brand_display: str
    price_num: Optional[float] = None
    rating_num: Optional[float] = None
    tags: Optional[str] = None
    buy_url: Optional[str] = None

class WishlistPayload(BaseModel):
    email: EmailStr
    products: List[Product]

# ------------------------------
# Endpoints
# ------------------------------
@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    """
    Return list of users without password for debugging / admin purposes.
    """
    users = db.query(User).all()
    return {
        "success": True,
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "username": u.username,
            }
            for u in users
        ],
    }


@router.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """
    Simple registration:
    - Normalisasi email
    - Cek duplikat
    - Hash password
    - Simpan ke DB

    Tetap return {success, message, user} supaya kompatibel dengan frontend.
    """
    email = user.email.strip().lower()
    username = user.username.strip()

    if not username:
        return {"success": False, "message": "Username is required"}

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return {"success": False, "message": "Email already registered"}

    user_obj = User(
        email=email,
        username=username,
        password_hash=hash_password(user.password),
    )
    db.add(user_obj)
    db.commit()
    db.refresh(user_obj)

    return {
        "success": True,
        "message": "Registration successful",
        "user": {
            "id": user_obj.id,
            "email": user_obj.email,
            "username": user_obj.username,
            "created_at": user_obj.created_at,
        },
    }


@router.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Simple login:
    - Normalisasi email
    - Ambil user
    - Verifikasi password hash

    Kalau gagal, tetap balikin {success: False, message: "..."} (tanpa HTTPException)
    supaya frontend bisa pakai pola `if (!json.success)`.
    """
    email = user.email.strip().lower()
    user_obj = db.query(User).filter(User.email == email).first()

    if not user_obj or not verify_password(user.password, user_obj.password_hash):
        return {"success": False, "message": "Invalid credentials"}

    return {
        "success": True,
        "message": "Login successful",
        "user": {
            "id": user_obj.id,
            "email": user_obj.email,
            "username": user_obj.username,
        },
    }


@router.post("/wishlist/add")
async def add_to_wishlist(payload: WishlistPayload, db: Session = Depends(get_db)):
    """
    Simpan banyak produk ke wishlist user.
    Dipanggil dari frontend dengan POST ke /auth/wishlist/add
    body: { email: string, products: [...] }
    """
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # optional: clear duplikasi
    for p in payload.products:
        exists = (
            db.query(WishlistItem)
            .filter(
                WishlistItem.user_id == user.id,
                WishlistItem.name_display == p.name_display,
                WishlistItem.brand_display == p.brand_display,
            )
            .first()
        )
        if exists:
            continue

        item = WishlistItem(
            user_id=user.id,
            product_id=p.id,
            name_display=p.name_display,
            brand_display=p.brand_display,
            image_url=p.image_url,
            price_num=p.price_num,
            rating_num=p.rating_num,
            tags=p.tags,
            buy_url=p.buy_url,
        )
        db.add(item)

    db.commit()

    return {"success": True, "message": "Products added to wishlist"}

# ====== GET WISHLIST (dipakai di /wishlist page) ======

@router.get("/wishlist")
async def get_wishlist(email: EmailStr, db: Session = Depends(get_db)):
    """
    GET /auth/wishlist?email=...
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return {"success": True, "wishlist": []}

    items = (
        db.query(WishlistItem)
        .filter(WishlistItem.user_id == user.id)
        .order_by(WishlistItem.created_at.desc())
        .all()
    )

    result = [
        {
            "id": item.id,
            "product_id": item.product_id,
            "name_display": item.name_display,
            "brand_display": item.brand_display,
            "image_url": item.image_url,
            "price_num": item.price_num,
            "rating_num": item.rating_num,
            "tags": item.tags,
            "buy_url": item.buy_url,
        }
        for item in items
    ]

    return {"success": True, "wishlist": result}

# ====== CLEAR WISHLIST ======

class ClearWishlistPayload(BaseModel):
    email: EmailStr

@router.post("/wishlist/clear")
async def clear_wishlist(payload: ClearWishlistPayload, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        return {"success": True, "message": "Nothing to clear"}

    db.query(WishlistItem).filter(WishlistItem.user_id == user.id).delete()
    db.commit()
    return {"success": True, "message": "Wishlist cleared"}