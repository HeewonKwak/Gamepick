from fastapi import APIRouter, HTTPException

from services.predict import gemini_ai
from models.prediction import (
    RecommendedGame,
    ModelInput
)

import time

router = APIRouter()


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_recommendations(data_point):
    return gemini_ai(data_point)


def make_list(gpt_answer):
    games = gpt_answer.strip().split("\n")
    return [game.split(". ")[1] for game in games]


@router.post(
    "/predict",
    response_model=RecommendedGame,
    name="predict:get-input",
)
async def gpt_recommend(data_input: ModelInput):

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        start_time = time.time()
        recommendations = get_recommendations(data_input)
        recommendations = make_list(recommendations)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"실행 시간: {elapsed_time:.4f} 초")

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return RecommendedGame(games=recommendations)