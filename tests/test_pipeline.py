import joblib
import pandas as pd

def test_model_predict():
    model = joblib.load("model.pkl")
    test_sample = pd.DataFrame([[
        14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28,
        2.29, 5.64, 1.04, 3.92, 1065.0
    ]])
    res = model.predict(test_sample)
    assert res[0] in [0,1,2]
