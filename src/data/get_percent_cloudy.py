import argparse
import numpy as np 
import pandas as pd 
import fsspec
from matplotlib import pyplot as plt 

try:
    from src.data.gcs_config import bucket_path
except ImportError:
    from gcs_config import bucket_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot cloud probability histograms for a collection of metadata CSVs.'
    )
    parser.add_argument(
        '--csv-glob',
        default=None,
        help='Explicit gs:// glob for metadata CSVs. Defaults to gs://<EDGESAT_GCS_BUCKET>/train/S2/*/*.csv.',
    )
    args = parser.parse_args()

    csv_path = args.csv_glob or bucket_path('train', 'S2', '*', '*.csv')
    fs = fsspec.filesystem("gs")
    csv_files = fs.glob(csv_path)

    dfs = []
    for fi in csv_files:
        print("Reading {}".format(fi))
        dfs.append(pd.read_csv("gs://" + fi))

    total_dfs = pd.concat(dfs)
    total_dfs["cloud_probability"].hist()
    plt.show()


if __name__ == "__main__":
    main()
