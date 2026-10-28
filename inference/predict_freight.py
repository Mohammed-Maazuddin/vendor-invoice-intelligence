'''
import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open (model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor Invoices.

    Parameters
    ----------
    input-data : dict

    returns
    -------
    pd.DataFrame with predicted freight cost
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    # example inference run (local testing)
    sample_data = {
        "Dollars" : [18500, 9000, 3000, 200],
        "Quantity" : [250, 100, 55,33]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
'''


import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Invoice dollars and quantity.

    Returns
    -------
    pd.DataFrame
        Input features, predicted freight, and warning.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    # Match the feature names and order used during training
    features = ["Dollars", "Quantity"]
    input_df = input_df[features]

    # Generate predictions
    predictions = model.predict(input_df)

    # Record negative predictions before applying the safeguard
    input_df["Prediction_Warning"] = predictions < 0

    # Prevent negative freight estimates in the displayed output
    input_df["Predicted_Freight"] = (
        pd.Series(predictions, index=input_df.index)
        .clip(lower=0)
        .round(2)
    )

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200],
        "Quantity": [250, 100, 55, 33]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
