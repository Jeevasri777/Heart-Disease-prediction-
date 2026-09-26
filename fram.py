# -----------------------------------------
# IMPORT LIBRARIES
# -----------------------------------------

import streamlit as st
import joblib
import pandas as pd



# -----------------------------------------
# LOAD DUMPED FILES
# -----------------------------------------

model = joblib.load(
    r"C:\Users\jeeva\OneDrive\Desktop\mini\heart.pkl"
)




# -----------------------------------------
# STREAMLIT PAGE
# -----------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)



st.title(
    "❤️ Heart Disease Prediction System"
)


st.write(
    "Enter Patient Details"
)



# -----------------------------------------
# USER INPUT
# -----------------------------------------

age = st.number_input(
    "Age"
)


sex = st.selectbox(
    "Sex (0/1)",
    [0,1]
)


cp = st.number_input(
    "Chest Pain Type",
    min_value=0,
    max_value=3
)


bp = st.number_input(
    "Blood Pressure"
)


chol = st.number_input(
    "Cholesterol"
)


fbs = st.selectbox(
    "FBS Over 120 (0/1)",
    [0,1]
)


ekg = st.number_input(
    "EKG Result"
)


max_hr = st.number_input(
    "Maximum Heart Rate"
)


ex_angina = st.selectbox(
    "Exercise Angina (0/1)",
    [0,1]
)


st_dep = st.number_input(
    "ST Depression"
)


slope = st.number_input(
    "Slope of ST"
)


vessels = st.number_input(
    "Number of Vessels"
)


thal = st.number_input(
    "Thallium"
)




# -----------------------------------------
# CREATE DATAFRAME
# -----------------------------------------


patient = pd.DataFrame(

    [[

        age,
        sex,
        cp,
        bp,
        chol,
        fbs,
        ekg,
        max_hr,
        ex_angina,
        st_dep,
        slope,
        vessels,
        thal

    ]],


    columns=[

        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium"

    ]

)



st.subheader(
    "Patient Details"
)


st.dataframe(
    patient
)




# -----------------------------------------
# PREDICTION
# -----------------------------------------


# -----------------------------------------
# PREDICTION
# -----------------------------------------

if st.button("Predict"):

    prediction = model.predict(
        patient
    )

    probability = model.predict_proba(
        patient
    )


    st.subheader(
        "Prediction Result"
    )


    if prediction[0] == 1:

        st.error(
            "⚠️ Heart Disease Detected"
        )

    else:

        st.success(
            "✅ No Heart Disease"
        )


    st.write(
        "Prediction Probability"
    )

    st.write(
        probability
    )