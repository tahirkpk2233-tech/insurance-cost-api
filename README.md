# Insurance Cost Prediction API

A machine learning API that predicts medical insurance charges from a person's details, built with FastAPI and scikit-learn.

## Features
- Linear Regression model trained on insurance data
- REST API endpoint: `POST /predict`
- Interactive docs at `/docs`

## Tech Stack
Python, FastAPI, scikit-learn, Pandas, Uvicorn

## How to Run
1. Install packages: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Start the API: `uvicorn main:app --reload`
4. Open `http://127.0.0.1:8000/docs` and test `/predict`

## Example Input
age: 30, sex: male, bmi: 25.5, children: 1, smoker: no, region: northwest

## Author
Tahir Ullah | BS Computer Science | Machine Learning
