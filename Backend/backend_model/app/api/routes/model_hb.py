from fastapi import APIRouter, HTTPException

from services.predict import HybridModel, HybridModel_Modify
from models.prediction import (
    RecommendedGame,
    ModelInput
)

import time
import joblib

router = APIRouter()


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_recommendations(data_point):
    hb_model = HybridModel(data_point)
    return hb_model.predict()

def get_recommendations2(data_point):
    # hb_model = joblib.load('tests/model.pkl')
    hb_model = HybridModel_Modify()
    # joblib.dump(hb_model, 'tests/model.pkl')
    return hb_model.predict(data_point)


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
        recommendations = get_recommendations2(data_input)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"실행 시간: {elapsed_time:.4f} 초")

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return RecommendedGame(games=recommendations)