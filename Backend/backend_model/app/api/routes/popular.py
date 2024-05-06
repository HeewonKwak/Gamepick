from fastapi import APIRouter, HTTPException

from services.predict import Most_popular_filter
from models.prediction import (
    RecommendedGame,
    ModelInput
)

import time

router = APIRouter()


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_recommendations(data_point):
    hb_model = Most_popular_filter(data_point)
    return hb_model.predict()


@router.post(
    "/predict",
    response_model=RecommendedGame,
    name="predict:get-input",
)
async def model_recommend(data_input: ModelInput):

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        start_time = time.time()
        recommendations = get_recommendations(data_input)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"실행 시간: {elapsed_time:.4f} 초")

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return RecommendedGame(games=recommendations)