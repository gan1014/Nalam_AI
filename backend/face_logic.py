
import os
import cv2
import numpy as np
import json
import base64
# from deepface import DeepFace  # Moved to lazy import inside functions
from cryptography.fernet import Fernet
from typing import List, Optional

# Load Secret Key for Encryption
SECRET_KEY = os.getenv("FACIAL_EMBEDDING_SECRET")
if not SECRET_KEY:
    # Fallback for dev environment, but should be set in .env
    SECRET_KEY = Fernet.generate_key().decode()

cipher_suite = Fernet(SECRET_KEY.encode())

FACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "face_data", "enrolled_faces")
MANUAL_REF_DIR = os.path.join(os.path.dirname(__file__), "face_data", "manual_reference")

def save_face_image(user_id: int, image_bytes: bytes) -> str:
    """
    Saves the enrolled face image locally for physical validation (captured via camera).
    """
    if not os.path.exists(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR, exist_ok=True)
        
    file_path = os.path.join(FACE_DATA_DIR, f"user_{user_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return file_path

def save_manual_reference(user_id: int, image_bytes: bytes) -> str:
    """
    Saves a manually uploaded image as the primary identity reference.
    """
    if not os.path.exists(MANUAL_REF_DIR):
        os.makedirs(MANUAL_REF_DIR, exist_ok=True)
        
    file_path = os.path.join(MANUAL_REF_DIR, f"ref_user_{user_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return file_path

def get_enrolled_image_path(user_id: int) -> Optional[str]:
    """
    Returns the path to the identity reference, prioritizing manual uploads and generic REF files.
    """
    # 1. Check for generic REF files first (highly user-friendly fallback)
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        ref_path = os.path.join(MANUAL_REF_DIR, f"REF{ext}")
        if os.path.exists(ref_path):
            return ref_path

    # 2. Check for user-specific manual uploads
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        manual_path = os.path.join(MANUAL_REF_DIR, f"ref_user_{user_id}{ext}")
        if os.path.exists(manual_path):
            return manual_path
        
    # 3. Fallback to automated enrollment folder
    enrolled_path = os.path.join(FACE_DATA_DIR, f"user_{user_id}.jpg")
    return enrolled_path if os.path.exists(enrolled_path) else None

def generate_embedding(image_data: bytes) -> Optional[np.ndarray]:
    """
    Detects face and generates a 128d or 512d embedding vector from raw image bytes.
    """
    try:
        # Prevent "lambda" layer name collision in TensorFlow
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Lazy import to speed up app load and prevent installation issues from blocking startup
        from deepface import DeepFace
        
        # DeepFace.represent returns a list of dictionaries (one per face)
        results = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
        
        if not results:
            return None
            
        # Extract the embedding from the first face detected
        embedding = results[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"❌ Embedding Generation Error: {e}")
        return None

def verify_against_file(live_bytes: bytes, user_id: int) -> dict:
    """
    Strictly compares a live capture against the saved reference file.
    Uses Facenet (already cached) with a TIGHT custom threshold to prevent false positives.
    """
    target_path = get_enrolled_image_path(user_id)
    if not target_path:
        return {"status": "NO_ENROLLED_FILE", "verified": False}

    # STRICT THRESHOLD for Facenet + cosine distance
    # Default Facenet cosine threshold is ~0.40. We use 0.30 for strictness.
    STRICT_THRESHOLD = 0.30

    try:
        # Prevent name collision
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except: pass

        # Load live image
        nparr = np.frombuffer(live_bytes, np.uint8)
        live_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Lazy import
        from deepface import DeepFace

        # DeepFace Verify with Facenet (already downloaded on this system)
        result = DeepFace.verify(
            live_img, 
            target_path, 
            model_name="Facenet",
            distance_metric="cosine",
            enforce_detection=True
        )
        
        distance = result["distance"]
        default_threshold = result["threshold"]
        
        # Apply OUR strict threshold instead of the model default
        is_verified = distance <= STRICT_THRESHOLD
        
        # Convert distance to similarity (0-1 scale, higher = better match)
        similarity = max(0, 1 - distance)
        
        print(f"🔍 VERIFICATION: distance={distance:.4f}, strict_threshold={STRICT_THRESHOLD}, default_threshold={default_threshold:.4f}, PASS={is_verified}")
        print(f"   Reference file: {target_path}")
        
        return {
            "status": "SUCCESS" if is_verified else "MISMATCH",
            "verified": is_verified,
            "similarity": similarity,
            "distance": distance,
            "threshold": STRICT_THRESHOLD
        }
    except Exception as e:
        print(f"❌ File-based verification failed: {e}")
        return {"status": "ERROR", "verified": False, "error": str(e)}

def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """
    Encrypts the embedding vector (as JSON string) using Fernet.
    """
    embedding_list = embedding.tolist()
    embedding_json = json.dumps(embedding_list)
    return cipher_suite.encrypt(embedding_json.encode())

def decrypt_embedding(encrypted_blob: bytes) -> Optional[np.ndarray]:
    """
    Decrypts the blob and returns the embedding as a numpy array.
    """
    try:
        decrypted_json = cipher_suite.decrypt(encrypted_blob).decode()
        embedding_list = json.loads(decrypted_json)
        return np.array(embedding_list)
    except Exception as e:
        print(f"❌ Embedding Decryption Error: {e}")
        return None

def compare_embeddings(live_embedding: np.ndarray, stored_embeddings: List[np.ndarray]) -> float:
    """
    Compares live embedding with multiple stored embeddings and returns the max similarity score (0 to 1).
    Uses cosine distance converted to similarity.
    """
    if not stored_embeddings:
        return 0.0
        
    similarities = []
    from scipy.spatial.distance import cosine
    
    for stored in stored_embeddings:
        # cosine distance is 0 for identical, 1 for orthogonal
        # similarity = 1 - distance
        dist = cosine(live_embedding, stored)
        similarities.append(1 - dist)
        
    return max(similarities) if similarities else 0.0
