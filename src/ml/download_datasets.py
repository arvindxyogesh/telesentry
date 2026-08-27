"""Optional supplementary dataset download.

FordA (engine acoustic/vibration sensor readings, UCR/UEA archive) is not
part of the core train/eval pipeline -- it has a different schema and
sampling regime than the CAN/IMU/GPS telemetry this project targets. It is
included as a real-world sensor time series that the same class of
reconstruction-based sequence models (LSTM autoencoder, attention detector)
could be adapted to in future work; see the README "Limitations & Future
Work" section.
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen


FORD_A_URL = "https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip"


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external datasets for telemetry anomaly project.")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    zip_path = out_dir / "FordA.zip"
    extract_dir = out_dir / "forda"

    print(f"Downloading FordA dataset: {FORD_A_URL}")
    download_file(FORD_A_URL, zip_path)
    print(f"Saved: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    files = sorted(str(p) for p in extract_dir.glob("*"))
    print("Extracted files:")
    for fp in files:
        print(f" - {fp}")


if __name__ == "__main__":
    main()
