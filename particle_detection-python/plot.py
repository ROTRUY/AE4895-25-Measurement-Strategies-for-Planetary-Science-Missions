import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from helper import request_csv_filepath

FIGURES_FOLDER: str = "./particle_detection-python/figures"


def plot_histogram(resolution):
    csv_path = request_csv_filepath("./csv")
    df = pd.read_csv(csv_path)

    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]

    logging.info("Available parameters:")
    for col in df.columns:
        print(f"   - {col}")
    param = input("\nEnter the parameter name to plot: ").strip()

    if param not in df.columns:
        raise ValueError(f"'{param}' is not a valid column name.")
    data = df[param].dropna()

    p1, p99 = data.quantile([0.01, 0.99])
    clipped_data = data[(data >= p1) & (data <= p99)]

    fig, ax = plt.subplots()
    plt.hist(clipped_data, bins=100)
    plt.ylabel("Frequency")
    plt.title(f"{param.replace('_', ' ').capitalize()} (N grains: {len(clipped_data)})")
    plt.grid(True)
    unit = r"$\mu m$" if resolution != "unknown" else "pixel"
    if param == 'area':
        plt.xlabel(rf"{param.replace('_', ' ').capitalize()} [{unit}$^2$]")
    elif param == 'orientation':
        # Convert to degrees from radians
        plt.xlabel(rf"{param.replace('_', ' ').capitalize()} [deg]")
    elif param.find('Intensity') < 0:
        # Any param without Intensity in its name
        plt.xlabel(rf"{param.replace('_', ' ').capitalize()} [{unit}]")
    else:
        plt.xlabel(rf"{param.replace('_', ' ').capitalize()}")
    # ax.set_yscale('log')
    # plt.xlim([30, max(clipped_data)])
    csv_path = Path(csv_path)
    histogram_path = f"{FIGURES_FOLDER}/{'_'.join(csv_path.with_suffix('').parts)}-{param}.png".replace("csv_", "")
    logging.info(f"Saving plot: {histogram_path}")
    plt.savefig(histogram_path)
    plt.show()
