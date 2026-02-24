
import bcrypt
import jwt
import datetime
import os
import random
import string
from typing import Optional

SECRET_KEY = os.getenv("JWT_SECRET", "tamil_nadu_health_dept_secure_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except:
        return None

def generate_captcha_text(length=5):
    # Exclusion list for high clarity: 0/O, 1/I, L, S/5, Z/2 sometimes confuse users
    chars = "ABCDEFGHJKMNPQRSTUVWXYZ346789"
    return ''.join(random.choices(chars, k=length))

def get_user_from_db(username: str):
    import sqlite3
    import os
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nalamai_local.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, username))
    user = cursor.fetchone()
    conn.close()
    return user
