from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from components.user_input import UserInput
from components.predict import predict_output
from components.prediction_response import PredictionResponse
from utils.logger import logger
from utils.exception import securityException
import uvicorn

app = FastAPI()

@app.get("/home")
def home():
    logger.info("Home endpoint called")
    return {"message": "Insurance premium prediction API"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: UserInput):
    logger.info("/predict endpoint called")
    try:
        user_features = {
            'bmi': data.bmi,
            'age_group': data.age_group,
            'lifestyle_risk': data.lifestyle_risk,
            'city_tier': data.city_tier,
            'income_lpa': data.income_lpa,
            'occupation': data.occupation
        }

        prediction = predict_output(user_features)
        return JSONResponse(status_code=200, content=prediction)

    except securityException as pe:
        logger.error(f"Prediction error: {pe}")
        raise HTTPException(status_code=400, detail=str(pe))
    except Exception as e:
        logger.exception("Unexpected error in prediction")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000,reload=True)# remove reload=True  Dockerization and Pushing to EC2 push . Use reload=True only in local development