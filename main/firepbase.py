import os, json
from firebase_admin import credentials, initialize_app
from google.cloud import firestore
import firebase_admin

def get_firestore_client():
    if not firebase_admin._apps:
        raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not raw:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")
        cred = credentials.Certificate(json.loads(raw))
        initialize_app(cred)
    return firestore.Client()
