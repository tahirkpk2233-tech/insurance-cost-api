import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("Medical Insurance Cost Predictor")
st.write("Enter your details to get an estimated yearly insurance charge.")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 60.0, 27.5)
children = st.slider("Number of children", 0, 10, 1)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox(
    "Region", ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict"):
    person = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }]
    )
    prediction = model.predict(person)[0]
    st.success(f"Estimated yearly charges: ${prediction:,.2f}")

st.caption("Linear Regression model trained on the insurance dataset. Estimate only.")