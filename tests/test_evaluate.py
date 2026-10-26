import pandas as pd

from src.evaluate import mae_by_group


#test 1 - tests that mae_by_group correctly calculates the MAE for each group
def test_mae_by_group():
    df = pd.DataFrame(
        {
            "hour": [1, 1, 2],
            "count": [10, 20, 5],
            "pred": [12, 18, 5],
        }
    )
    result = mae_by_group(df, "pred", "hour")
    assert result.loc[1] == 2
    assert result.loc[2] == 0
