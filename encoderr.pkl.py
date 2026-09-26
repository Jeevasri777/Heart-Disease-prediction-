import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder


# Load dataset

df = pd.read_csv("datasetss.csv")


# Store encoders

encoder = {}


# Encode text columns

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col]
    )

    encoder[col] = le



# Save encoder

joblib.dump(
    encoder,
    "encoder.pkl"
)


print("encoder.pkl created")