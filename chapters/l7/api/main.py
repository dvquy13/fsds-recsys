import json
from typing import Optional

import redis
from fastapi import FastAPI, HTTPException, Query

from .load_examples import custom_openapi

app = FastAPI()

# Initialize Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
redis_output_i2i_key_prefix = "output:i2i:"
redis_feature_recent_items_key_prefix = "feature:user:recent_items:"
redis_output_popular_key = "output:popular"

# Set the custom OpenAPI schema with examples
app.openapi = lambda: custom_openapi(
    app,
    redis_client,
    redis_output_i2i_key_prefix,
    redis_feature_recent_items_key_prefix,
)


@app.get("/recs/i2i")
def get_recommendations_i2i(
    item_id: str = Query(
        ..., description="ID of the item to get recommendations for"
    ),  # ... denotes required param in FastAPI
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
):
    # Step 1: Get recommendations for the item ID from Redis
    recommendations_key = redis_output_i2i_key_prefix + item_id
    rec_data = redis_client.get(recommendations_key)

    if not rec_data:
        raise HTTPException(
            status_code=404, detail=f"No recommendations found for item_id: {item_id}"
        )

    # Parse the stored recommendation data
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])

    # Step 2: Limit the output by count if count is provided
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]

    # Step 3: Format and return the output
    return {
        "item_id": item_id,
        "recommendations": {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores},
    }


@app.get("/recs/u2i")
def get_recommendations_u2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
):
    # Step 1: Get the recent items for the user
    recent_items_key = redis_feature_recent_items_key_prefix + user_id
    item_ids_str = redis_client.get(recent_items_key)

    if not item_ids_str:
        raise HTTPException(
            status_code=404, detail=f"No recent items found for user_id: {user_id}"
        )

    # Step 2: Split the item IDs string by "__"
    item_ids = item_ids_str.split("__")

    # Step 3: Get the most recently interacted item ID
    last_item_id = item_ids[-1]

    # Step 4: Call the i2i endpoint internally to get recommendations for that item
    recommendations = get_recommendations_i2i(last_item_id, count)

    # Step 5: Format and return the output
    return {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recommendations["recommendations"],
    }


@app.get("/recs/popular")
def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return")
):
    # Step 1: Get popular recommendations from Redis
    rec_data = redis_client.get(redis_output_popular_key)

    if not rec_data:
        raise HTTPException(status_code=404, detail="No popular recommendations found")

    # Parse the stored recommendation data
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])

    # Step 2: Limit the output by count if count is provided
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]

    # Step 3: Format and return the output
    return {
        "recommendations": {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores},
    }
