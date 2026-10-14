import pandas as pd

from src.data import load_data

#sample dataset
SAMPLE = """instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
1,2011-01-01,1,0,1,0,0,6,0,1,0.24,0.2879,0.81,0.0,3,13,16
2,2011-01-01,1,0,1,1,0,6,0,1,0.22,0.2727,0.80,0.0,8,32,40
3,2011-01-01,1,0,1,2,0,6,0,1,0.22,0.2727,0.80,0.0,5,27,32
"""

#test function - writes data into temporary file and loads using load_data function
def test_load_data(tmp_path):
    csv = tmp_path /"hour.csv"
    csv.write_text(SAMPLE)
    df = load_data(csv)


    #check casual & registered columns removed
    assert "casual" not in df.columns
    assert "registered" not in df.columns
    #check data is sorted by datetime
    assert df["datetime"].is_monotonic_increasing
    #check correct timestamp for 2nd row
    assert df["datetime"].iloc[1] == pd.Timestamp("2011-01-01 01:00")
    #check rentals count matches expected values
    assert df["count"].tolist() == [16, 40, 32]
