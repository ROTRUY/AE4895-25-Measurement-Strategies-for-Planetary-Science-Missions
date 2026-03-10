import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def read_spectrum(filepath: str):
    """
    Reads Raman spectroscopy data from a text file.
    
    :param filepath: Path to the text file containing the spectrum data.
    :type filepath: str
    :return: Tuple of angle and intensity arrays.
    """
    data = np.loadtxt(filepath, comments='#')
    
    angle = data[:, 0]
    intensity = data[:, 1]
    
    return angle, intensity

def detect_peaks(angle, intensity, prominence=200, distance=1):
    peaks, _ = find_peaks(intensity, prominence=prominence, distance=distance)

    return angle[peaks], intensity[peaks]

# Read data
AEMA01_angle, AEMA01_intensity = read_spectrum(r"XRD\\mars_analogue.txt")    # Martian dust analogue
AEFE01_angle, AEFE01_intensity = read_spectrum(r"XRD\\aefe01.txt")  # Hematite reference
AEFE02_angle, AEFE02_intensity = read_spectrum(r"XRD\\aefe02.txt")  # Magnetite reference

# Get peaks
AEMA01_peaks = detect_peaks(AEMA01_angle, AEMA01_intensity, 45, 1)
AEFE01_peaks = detect_peaks(AEFE01_angle, AEFE01_intensity, 200, 1)
AEFE02_peaks = detect_peaks(AEFE02_angle, AEFE02_intensity, 200, 1)

# Define the spectra data and metadata for plotting
to_plot = [
    ("AEMA01", AEMA01_angle, AEMA01_intensity, "Global Martian Dust Analogue", "blue", AEMA01_peaks),
    ("AEFE01", AEFE01_angle, AEFE01_intensity, "Hematite",  "red", AEFE01_peaks),
    ("AEFE02", AEFE02_angle, AEFE02_intensity, "Magnetite", "black", AEFE02_peaks)
]

# Individual Plots
for item in to_plot:
    plt.figure(figsize=(10, 6))
    plt.plot(item[1], item[2], label=item[3], color=item[4])
    # Mark peaks
    peakColour = "brown"
    plt.scatter(*item[5], color=peakColour, marker='x')
    for wn, tr in zip(*item[5]):
        plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)

    plt.xlabel('Angle (°)')
    plt.ylabel('Intensity')
    plt.title(f"XRD Spectrum of {item[3]} with peaks")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"XRDPlots\{item[0]}_with_peaks.png", dpi=400)
    plt.close()

# Plot all spectra together
plt.figure(figsize=(10, 6))
plt.plot(AEMA01_angle, AEMA01_intensity, label="Global Martian Dust Analogue", color="blue")
plt.plot(AEFE01_angle, AEFE01_intensity, label="Hematite", color="red")
plt.plot(AEFE02_angle, AEFE02_intensity, label="Magnetite", color="black")
peakColour = "black"
plt.scatter(*AEMA01_peaks, color=peakColour, marker='x')
plt.scatter(*AEFE01_peaks, color=peakColour, marker='x')
plt.scatter(*AEFE02_peaks, color=peakColour, marker='x')
for wn, tr in zip(*AEMA01_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
for wn, tr in zip(*AEFE01_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
for wn, tr in zip(*AEFE02_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
plt.xlabel('Angle (°)')
plt.ylabel('Intensity')
plt.title("XRD Spectra of Martian Dust Analogue and References with peaks")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("XRDPlots\All_Spectra_with_peaks.png", dpi=400)