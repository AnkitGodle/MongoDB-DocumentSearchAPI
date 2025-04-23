"""
This module  is the main entry point for the FastAPI application.
It initializes the FastAPI app and defines the API endpoints for movie search and management.
It includes endpoints for searching movies, creating new movie documents,
updating existing movies, and deleting movies.
It also handles exceptions and returns appropriate HTTP responses.
app.main.py
"""

from fastapi import FastAPI, HTTPException
from typing import List
from app.movie_service import *

app = FastAPI()

@app.post("/movies/search", response_model=List[SearchResult])
def search(query: SearchQuery):
    try:
        result = search_documents(query)
        if not result:
            raise HTTPException(status_code=404, detail="No results found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/movies/create")
def create(movie: MovieCreate):
    return create_movie_doc(movie)

@app.get("/movies/get/{movie_id}")
def get(movie_id: str):
    movie = get_movie_doc(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.put("/movies/update/{movie_id}")
def update(movie_id: str, movie: MovieUpdate):
    success = update_movie_doc(movie_id, movie)
    if not success:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"message": "Movie updated"}

@app.delete("/movies/delete/{movie_id}")
def delete(movie_id: str):
    success = delete_movie_doc(movie_id)
    if not success:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"message": "Movie deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)
