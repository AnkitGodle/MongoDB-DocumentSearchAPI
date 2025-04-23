"""
This module handles the connection to the MongoDB database.
It provides functions to get the source and target collections
and to retrieve the MongoDB collections.
app.db.py
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")
SOURCE_COLLECTION = os.getenv("SOURCE_COLLECTION_NAME")
TARGET_COLLECTION = os.getenv("TARGET_COLLECTION_NAME")

EMBEDDING_MODEL_PATH = "app/models/all-mpnet-base-v2"

_client = MongoClient(MONGO_URI)
_db = _client[DB_NAME]

def get_source_collection():
    return _db[SOURCE_COLLECTION]

def get_target_collection():
    return _db[TARGET_COLLECTION]

def get_mongo_collections():
    return get_source_collection(), get_target_collection()
