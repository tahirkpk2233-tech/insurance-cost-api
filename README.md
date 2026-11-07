# Medical Insurance Cost Predictor

A machine learning project that estimates yearly medical insurance
charges from a person's details, served as a FastAPI service and a
Streamlit web app.

## Model
- Linear Regression inside a scikit-learn Pipeline
  (OneHotEncoder for categorical features)
- R2 score of about 0.78 on a held-out 20% test set
- Dataset: 1,338 records (age, sex, BMI, children, smoker, region)

## Files
- train.py : trains the model and saves model.joblib
- main.py  : FastAPI service with input validation (/predict)
- app.py   : Streamlit web interface
- insurance.csv : dataset

## Run locally
pip install -r requirements.txt
python train.py
streamlit run app.py

To run the API instead:
uvicorn main:app --reload
then open http://127.0.0.1:8000/docs

## Notes
- Smoking status has the largest effect on predicted cost.
- This is an educational estimate, not real insurance pricing.
