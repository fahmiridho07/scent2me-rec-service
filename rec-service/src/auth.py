from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
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


@router.post("/wishlist/{user_email}/add")
async def add_to_wishlist(
    user_email: str,
    products: List[Product],
    db: Session = Depends(get_db),
):
    """
    Simpan produk ke wishlist berdasarkan email user.
    Struktur body: langsung list Product (bukan dibungkus object lain).
    """
    user_obj = db.query(User).filter(User.email == user_email).first()
    if not user_obj:
        return {"success": False, "message": "User not found"}

    created_count = 0

    for p in products:
        # Cegah duplikat basic (berdasarkan user + nama + brand)
        exists = (
            db.query(WishlistItem)
            .filter(
                WishlistItem.user_id == user_obj.id,
                WishlistItem.name_display == p.name_display,
                WishlistItem.brand_display == p.brand_display,
            )
            .first()
        )
        if exists:
            continue

        item = WishlistItem(
            user_id=user_obj.id,
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
        created_count += 1

    db.commit()

    return {
        "success": True,
        "message": f"Products added to wishlist ({created_count} new items).",
    }


@router.get("/wishlist/{user_email}")
async def get_wishlist(user_email: str, db: Session = Depends(get_db)):
    """
    Optional: ambil wishlist untuk user tertentu.
    Bisa dipakai page /wishlist di frontend.
    """
    user_obj = db.query(User).filter(User.email == user_email).first()
    if not user_obj:
        return {"success": False, "message": "User not found", "items": []}

    items = (
        db.query(WishlistItem)
        .filter(WishlistItem.user_id == user_obj.id)
        .order_by(WishlistItem.created_at.desc())
        .all()
    )

    return {
        "success": True,
        "items": [
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
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ],
    }