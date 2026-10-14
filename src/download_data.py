"""Download the UCI Bike Sharing dataset into dat/raw/."""

import io
import urllib.request
import zipfile
from pathlib import Path

#url stores download address and RAW_DIR stores folder where dataset saved
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_DIR = Path("data/raw")

#function called download , creates destination folder if it doesnt exist
def download(dest: Path = RAW_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    # downloads hour.csv into dest - skips if already exists
    target = dest / "hour.csv"
    if target.exists():
        return target
    #connects to URL, downloads ZIP into memory and opens it
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        #copies hour.csv from ZIP into data/raw folder and returns path
        with archive.open("hour.csv") as src, open(target, "wb") as out:
            out.write(src.read())
        return target


#when file run directly - python calls download() and prints saved location 
if __name__ == "__main__":
    print(f"Saved to {download()}")


