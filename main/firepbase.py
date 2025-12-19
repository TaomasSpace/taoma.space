import os, json
from firebase_admin import credentials, initialize_app, firestore as admin_firestore
import firebase_admin

def get_firestore_client():
    """
    Returns a Firestore client using the service account JSON provided via
    FIREBASE_SERVICE_ACCOUNT_JSON. This uses firebase_admin's firestore client
    so the given credentials are honored (no reliance on ADC on the host).
    """
    if not firebase_admin._apps:
        raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not raw:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")
        cred = credentials.Certificate(json.loads(raw))
        initialize_app(cred)
    return admin_firestore.client()
