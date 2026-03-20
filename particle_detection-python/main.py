import logging

from plot import plot_histogram
from threshold import threshold_grain_detection
from watershed import watershed_grain_detection

logging.basicConfig(
    level=logging.INFO,  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    threshold_grain_detection(image_name="AEQ01.jpeg", resolution="unknown", opening_radius=20)
    watershed_grain_detection(image_name="AEQ01.jpeg", resolution="unknown", opening_radius=20)
    plot_histogram(resolution="unknown")
