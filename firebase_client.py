"""
Firebase client for state management and real-time data streaming.
Primary database for the Evolution Ecosystem per mission constraints.
"""
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore, db
from google.cloud.firestore_v1 import Client as FirestoreClient
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.exceptions import GoogleCloudError

from config import config

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Firebase client for state management and real-time data"""