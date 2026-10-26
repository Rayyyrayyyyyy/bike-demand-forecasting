import pandas as pd

from src.train import baseline_predict, evaluate, time_split


#creating a small dataframe to use in the tests
def make_df():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2012-06-30 23:00", "2012-07-01 00:00", "2012-07-01 01:00"]
            ),
            "hour": [23, 0, 1],
            "workingday": [0, 0, 0],
            "count": [10, 20, 30],
        }
    )

#test 1 - test the time based split correctly separates test and training data
def test_time_split_has_no_overlap():
    train, test = time_split(make_df())
    assert len(train) == 1
    assert len(test) == 2
    assert train["datetime"].max() < test["datetime"].min()

#test 2 - tests that the evaluation function doesn't return errors when predictions match actual values 
def test_evaluate_perfect_prediction():
    result = evaluate([1, 2, 3], [1, 2, 3])
    assert result["mae"] == 0
    assert result["rmse"] == 0

#test 3 - tests that the basline model predicts the average
def test_baseline_uses_training_means():
    df = pd.DataFrame(
        {
            "hour": [8, 8, 8],
            "workingday": [1, 1, 1],
            "count": [10, 30, 0],
        }
    )
    train, test = df.iloc[:2], df.iloc[2:]
    assert baseline_predict(train, test).tolist() == [20.0]