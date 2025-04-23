"""
This  module is responsible for ingesting new movie data into the database.
It connects to the source and target MongoDB collections, processes the documents,
and generates embeddings for the movie plots.
app.ingest.py
"""
from tqdm import tqdm
from app.db import get_mongo_collections
from app.movie_service import generate_embedding
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("app/embed_model/all-mpnet-base-v2")
def preprocess_document(doc):
    # Replace empty fullplot with plot
    if not doc.get("fullplot") or not doc["fullplot"].strip():
        doc["fullplot"] = doc.get("plot", "")
    
    # Convert list fields to comma-separated strings
    # for key in ["cast", "genres", "directors", "countries"]:
    #     if key in doc and isinstance(doc[key], list):
    #         doc[key] = ", ".join(doc[key])
    
    return doc

def extract_text(doc):
    doc = preprocess_document(doc)
    # return "\n".join([f"{k}: {doc[k]}" for k in keys_to_extract if k in doc])
    return doc.get("fullplot", "")  

def ingest_new_collection(limit=5000):
    source_collection, target_collection = get_mongo_collections()

    cursor = source_collection.find({
        "plot": {"$exists": True},
        # "fullplot": {"$exists": True}
    }).limit(limit)

    count = 0

    for doc in tqdm(cursor, total=limit, desc="Processing documents"):
        if target_collection.find_one({"_id": doc["_id"]}):
            print(f"[{count}] Skipped duplicate: {doc.get('title', 'Untitled')}")
            continue
        text = extract_text(doc)
        # tokens = len(tokenizer.encode(text, truncation=False))
        # print(tokens)
        # token_count.append(tokens)

        if not text.strip():
            print(f"[{count}] Skipped empty document")
            count += 1
            continue
        
        doc["embedding"] = generate_embedding(text)
        target_collection.insert_one(doc)
        
        # print(f"[{count}] Inserted: {doc.get('title', 'Untitled')} Count:{tokens}")
        print(f"[{count}] Inserted: {doc.get('title', 'Untitled')}")

if __name__ == "__main__":
    ingest_new_collection()
