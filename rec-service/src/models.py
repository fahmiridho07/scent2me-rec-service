# src/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    wishlist_items = relationship("WishlistItem", back_populates="user", cascade="all, delete-orphan")


class WishlistItem(Base):
    __tablename__ = "wishlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # simpan basic product info
    product_id = Column(String(100), nullable=True)
    name_display = Column(String(255), nullable=False)
    brand_display = Column(String(255), nullable=False)
    image_url = Column(Text, nullable=True)
    price_num = Column(Float, nullable=True)
    rating_num = Column(Float, nullable=True)
    tags = Column(Text, nullable=True)
    buy_url = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="wishlist_items")
