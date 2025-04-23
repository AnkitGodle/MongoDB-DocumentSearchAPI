"""This module serves as a service layer for the movie database, providing
functions to create, read, update, and delete movie documents in the database.
It also includes functionality for searching movies using hybrid, text and vector search
It uses the SentenceTransformer model for generating embeddings and MongoDB
for storing and querying movie data.
app.movie_service.py
"""
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import List, Optional

from app.db import get_target_collection
from sentence_transformers import SentenceTransformer

collection = get_target_collection()
model = SentenceTransformer("all-mpnet-base-v2") #you can downloadand save to a path if needed

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
    year_gt: Optional[int] = None
    genre: Optional[str] = None
    type: Optional[str] = "vector"  # "vector" or "hybrid" or "text"

class SearchResult(BaseModel):
    id: Optional[str] = Field(alias="_id")
    title: str
    fullplot: str
    writers: Optional[List[str]] = None
    cast: Optional[List[str]] = None
    rated: Optional[str] = None
    genres: Optional[List[str]] = None
    year: Optional[int] = None
    score: float
    class Config:
        validate_by_name = True
        
class MovieCreate(BaseModel):
    title: str
    fullplot: str

class MovieUpdate(BaseModel):
    title: Optional[str] = None
    fullplot: Optional[str] = None


def generate_embedding(text: str):
    return model.encode(text, normalize_embeddings=True).tolist()

def build_pipeline(query: SearchQuery, embedding):
    filters = {}
    if query.year_gt:
        filters["year"] = {"$gt": query.year_gt}
    if query.genre:
        filters["genres"] = query.genre
    

    vector_pipeline = [
            {"$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "filter": filters,
                "queryVector": embedding,
                "numCandidates": query.top_k * 20,
                "limit": query.top_k
            }},
            {"$project": {
                "_id": 1,
                "title": 1, "fullplot": 1, "writers": 1, "cast": 1,
                "rated": 1, "genres": 1, "year": 1,
                "score": {"$meta": "vectorSearchScore"}
            }}]
    
    text_pipeline = [
            {"$search": {
                "index": "default",
                "text": {
                    "query": query.query,
                    "path": ["fullplot", "genres", "title"],
                    "fuzzy": {"maxEdits": 2}
                }
            }},
            {"$match": filters} if filters else {},
            {"$project": {
                "_id": 1,
                "title": 1, "fullplot": 1, "writers": 1, "cast": 1,
                "rated": 1, "genres": 1, "year": 1,
                "score": {"$meta": "searchScore"}
            }},
            {"$limit": query.top_k}
        ]
        
    if query.type == "vector":
        return vector_pipeline

    elif query.type == "text":
        return [stage for stage in text_pipeline if stage]

    raise ValueError("Invalid search type")

def hybrid_search(query: SearchQuery, embedding):
    # Run both pipelines
    vector_pipeline = build_pipeline(SearchQuery(**{**query.model_dump(), "type": "vector"}), embedding)
    text_pipeline = build_pipeline(SearchQuery(**{**query.model_dump(), "type": "text"}), embedding)

    vector_results = list(collection.aggregate(vector_pipeline))
    text_results = list(collection.aggregate(text_pipeline))

    # Normalize scores from each result set
    vector_scores = normalize_scores(vector_results, "score")
    text_scores = normalize_scores(text_results, "score")

    result_map = {}

    # Weightages 
    vector_weight = 0.6
    text_weight = 0.4

    # Merge vector results
    for _id, v in vector_scores.items():
        result_map[_id] = {"doc": v["doc"], "score": v["score"] * vector_weight}

    # Merge text results
    for _id, t in text_scores.items():
        if _id in result_map:
            result_map[_id]["score"] += t["score"] * text_weight
        else:
            result_map[_id] = {"doc": t["doc"], "score": t["score"] * text_weight}

    # Sort by combined score
    sorted_docs = sorted(result_map.values(), key=lambda x: x["score"], reverse=True)

    return [SearchResult(
        _id=str(d["doc"].get("_id")),
        title=d["doc"].get("title", ""),
        fullplot=d["doc"].get("fullplot", ""),
        genres=d["doc"].get("genres", []),
        year=d["doc"].get("year"),
        score=d["score"],
        writers=d["doc"].get("writers", []),
        cast=d["doc"].get("cast", []),
        rated=d["doc"].get("rated", "")
    ) for d in sorted_docs[:query.top_k]]

def search_documents(query: SearchQuery):
    embedding = generate_embedding(query.query)
    
    if query.type == "hybrid":
        return hybrid_search(query, embedding)
    
    pipeline = build_pipeline(query, embedding)
    results = collection.aggregate(pipeline)
    return [SearchResult(
        _id=str(doc.get("_id")),
        title=doc.get("title", ""),
        fullplot=doc.get("fullplot", ""),
        genres=doc.get("genres", []),
        year=doc.get("year"),
        score=doc.get("score", 0.0),
        writers=doc.get("writers", []),
        cast=doc.get("cast", []),
        rated=doc.get("rated", "")
    ) for doc in results]

def normalize_scores(results, score_field):
    if not results:
        return {}

    scores = [r.get(score_field, 0.0) for r in results]
    min_score, max_score = min(scores), max(scores)
    range_score = max_score - min_score + 1e-6  # to avoid division by zero

    normalized = {}
    for doc in results:
        doc_id = doc["_id"]
        raw_score = doc.get(score_field, 0.0)
        normalized_score = (raw_score - min_score) / range_score
        normalized[doc_id] = {"doc": doc, "score": normalized_score}
    return normalized

def create_movie_doc(movie):
    doc = movie.model_dump()
    doc["embedding"] = generate_embedding(doc["fullplot"])
    result = collection.insert_one(doc)
    return {"id": str(result.inserted_id)}

def get_movie_doc(movie_id: str):
    doc = collection.find_one({"_id": ObjectId(movie_id)})
    if not doc:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc

def update_movie_doc(movie_id: str, movie):
    update_data = movie.model_dump(exclude_unset=True)
    if "fullplot" in update_data:
        update_data["embedding"] = generate_embedding(update_data["fullplot"])
    result = collection.update_one({"_id": ObjectId(movie_id)}, {"$set": update_data})
    return result.matched_count > 0

def delete_movie_doc(movie_id: str):
    result = collection.delete_one({"_id": ObjectId(movie_id)})
    return result.deleted_count > 0
