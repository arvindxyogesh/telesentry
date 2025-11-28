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
