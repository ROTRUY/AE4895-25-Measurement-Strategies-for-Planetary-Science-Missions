import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from pybaselines import Baseline

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

def read_spectrum_csv(filepath: str):
    """
    Reads Raman spectroscopy data from a text file.
    
    :param filepath: Path to the text file containing the spectrum data.
    :type filepath: str
    :return: Tuple of angle and intensity arrays.
    """
    data = np.loadtxt(filepath, delimiter=",")
    
    angle = data[:, 0]
    intensity = data[:, 1]
    
    return angle, intensity

def detect_peaks(angle, intensity, prominence=0.3, distance=1):
    peaks, _ = find_peaks(intensity, prominence=prominence, distance=distance)

    return angle[peaks], intensity[peaks]

def baseline_correct(angle, intensity):
    baseline_fitter = Baseline(x_data=angle)
    baseline, params = baseline_fitter.asls(intensity, lam=1e7)
    corrected = intensity - baseline
    return corrected

def normalize(arr):
    return arr / np.max(arr)

### READ DATA ###
# Martian Dust Analogue and Reference Spectra 
AEMA01_angle, AEMA01_intensity = read_spectrum(r"XRD\\AEMA01.txt")  # Martian dust analogue
AEMA01_intensity = baseline_correct(AEMA01_angle, AEMA01_intensity) # Baseline correction
AEMA01_intensity = savgol_filter(AEMA01_intensity, 21, 3)           # Smooth the spectrum using Savitzky-Golay filter
AEMA01_intensity = normalize(AEMA01_intensity)                      # Normalize intensity

AEFE01_angle, AEFE01_intensity = read_spectrum(r"XRD\\AEFE01.txt")  # Hematite reference
AEFE01_intensity = baseline_correct(AEFE01_angle, AEFE01_intensity) # Baseline correction
AEFE01_intensity = savgol_filter(AEFE01_intensity, 21, 3)           # Smooth the spectrum using Savitzky-Golay filter
AEFE01_intensity = normalize(AEFE01_intensity)                      # Normalize intensity

AEFE02_angle, AEFE02_intensity = read_spectrum(r"XRD\\AEFE02.txt")  # Magnetite reference
AEFE02_intensity = baseline_correct(AEFE02_angle, AEFE02_intensity) # Baseline correction
AEFE02_intensity = savgol_filter(AEFE02_intensity, 21, 3)           # Smooth the spectrum using Savitzky-Golay filter
AEFE02_intensity = normalize(AEFE02_intensity)                      # Normalize intensity

# G11 Analogue Spectra for cross-checking
AEMA01_angle_11, AEMA01_intensity_11 = read_spectrum(r"XRD\\AEMA01_11.txt")  # Martian dust analogue of G11, for cross-checking
AEMA01_intensity_11 = baseline_correct(AEMA01_angle_11, AEMA01_intensity_11) # Baseline correction
AEMA01_intensity_11 = savgol_filter(AEMA01_intensity_11, 21, 3)              # Smooth the spectrum using Savitzky-Golay filter
AEMA01_intensity_11 = normalize(AEMA01_intensity_11)                         # Normalize intensity

AEMA02_angle_11, AEMA02_intensity_11 = read_spectrum(r"XRD\\AEMA02_11.txt")  # Jezero dust analogue of G11, for cross-checking
AEMA02_intensity_11 = baseline_correct(AEMA02_angle_11, AEMA02_intensity_11) # Baseline correction
AEMA02_intensity_11 = savgol_filter(AEMA02_intensity_11, 21, 3)              # Smooth the spectrum using Savitzky-Golay filter
AEMA02_intensity_11 = normalize(AEMA02_intensity_11)                         # Normalize intensity

# RUFF Reference Spectra for cross-checking
AEFE01_RUFF_angle, AEFE01_RUFF_intensity = read_spectrum_csv(r"XRD\\RUFF_AEFE01_alt.txt")  # Hematite reference from RUFF database
AEFE01_RUFF_intensity = baseline_correct(AEFE01_RUFF_angle, AEFE01_RUFF_intensity) # Baseline correction
AEFE01_RUFF_intensity = savgol_filter(AEFE01_RUFF_intensity, 21, 3)                # Smooth the spectrum using Savitzky-Golay filter
AEFE01_RUFF_intensity = normalize(AEFE01_RUFF_intensity)                           # Normalize intensity

AEFE02_RUFF_angle, AEFE02_RUFF_intensity = read_spectrum_csv(r"XRD\\RUFF_AEFE02.txt")  # Magnetite reference from RUFF database
AEFE02_RUFF_intensity = baseline_correct(AEFE02_RUFF_angle, AEFE02_RUFF_intensity) # Baseline correction
AEFE02_RUFF_intensity = savgol_filter(AEFE02_RUFF_intensity, 21, 3)                # Smooth the spectrum using Savitzky-Golay filter
AEFE02_RUFF_intensity = normalize(AEFE02_RUFF_intensity)                           # Normalize intensity

### GET PEAKS ###
# Martian Dust Analogue and References
AEMA01_peaks = detect_peaks(AEMA01_angle, AEMA01_intensity, 0.3)
AEFE01_peaks = detect_peaks(AEFE01_angle, AEFE01_intensity, 0.1)
AEFE02_peaks = detect_peaks(AEFE02_angle, AEFE02_intensity, 0.08)

# G11 Analogues
AEMA01_11_peaks = detect_peaks(AEMA01_angle_11, AEMA01_intensity_11, 0.15)
AEMA02_11_peaks = detect_peaks(AEMA02_angle_11, AEMA02_intensity_11, 0.4)

# RUFF References
AEFE01_RUFF_peaks = detect_peaks(AEFE01_RUFF_angle, AEFE01_RUFF_intensity, 0.15)
AEFE02_RUFF_peaks = detect_peaks(AEFE02_RUFF_angle, AEFE02_RUFF_intensity, 0.11)


### INDIVIDUAL PLOTTING ###
# Define list of spectra to plot, with their respective angles, intensities, labels, colors, and peaks
to_plot = [
    ("AEMA01", AEMA01_angle, AEMA01_intensity, "Global Martian Dust Analogue (AEMA01)", "blue", AEMA01_peaks),
    ("AEFE01", AEFE01_angle, AEFE01_intensity, "Hematite (AEFE01)",  "red", AEFE01_peaks),
    ("AEFE02", AEFE02_angle, AEFE02_intensity, "Magnetite (AEFE02)", "black", AEFE02_peaks),
    ("AEMA01_11", AEMA01_angle_11, AEMA01_intensity_11, "Martian Dust Analogue G11 (AEMA01)", "orange", AEMA01_11_peaks),
    ("AEMA02_11", AEMA02_angle_11, AEMA02_intensity_11, "Jezero Dust Analogue G11 (AEMA02)", "purple", AEMA02_11_peaks)
]

# Individual Plots
for item in to_plot:
    plt.figure(figsize=(14, 6))
    plt.plot(item[1], item[2], label=item[3], color=item[4])
    # Mark peaks
    peakColour = "brown"
    plt.scatter(*item[5], color=peakColour, marker='x')
    for wn, tr in zip(*item[5]):
        plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)

    plt.xlabel(f'2{chr(977)} (°)')
    plt.ylabel('Normalized Intensity (counts)')
    plt.title(f"Normalized XRD Spectrum of {item[3]} with peaks")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"XRDPlots\\{item[0]}_WP_N.png", dpi=500)
    plt.close()


### PLOT ALL SPECTRA TOGETHER ###
# Without G11 spectra
plt.figure(figsize=(10, 6))
plt.plot(AEMA01_angle, AEMA01_intensity, label="Global Martian Dust Analogue (AEMA01)", color="blue")
plt.plot(AEFE01_angle, AEFE01_intensity, label="Hematite (AEFE01)", color="red")
plt.plot(AEFE02_angle, AEFE02_intensity, label="Magnetite (AEFE02)", color="black")
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
plt.xlabel(f'2{chr(977)} (°)')
plt.ylabel('Normalized Intensity (counts)')
plt.title("Normalized XRD Spectra of Martian Dust Analogue and References with peaks")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("XRDPlots\\All_Spectra_WP_N.png", dpi=500)
plt.close()

# With G11 spectra
plt.figure(figsize=(10, 6))
plt.plot(AEMA01_angle, AEMA01_intensity, label="Global Martian Dust Analogue", color="blue")
plt.plot(AEMA01_angle_11, AEMA01_intensity_11, label="Martian Dust Analogue G11", color="orange")
plt.plot(AEFE01_angle, AEFE01_intensity, label="Hematite", color="red")
plt.plot(AEFE02_angle, AEFE02_intensity, label="Magnetite", color="black")
peakColour = "black"
plt.scatter(*AEMA01_peaks, color=peakColour, marker='x')
plt.scatter(*AEMA01_11_peaks, color=peakColour, marker='x')
plt.scatter(*AEFE01_peaks, color=peakColour, marker='x')
plt.scatter(*AEFE02_peaks, color=peakColour, marker='x')
for wn, tr in zip(*AEMA01_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
for wn, tr in zip(*AEMA01_11_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
for wn, tr in zip(*AEFE01_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
for wn, tr in zip(*AEFE02_peaks):
    plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
plt.xlabel(f'2{chr(977)} (°)')
plt.ylabel('Normalized Intensity (counts)')
plt.title("Normalized XRD Spectra of Martian Dust Analogue and References with peaks")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("XRDPlots\\All_Spectra_WP_11_N.png", dpi=500)
plt.close()

### STACKED PLOTS FOR COMPARISON ###
# Hematite + Magnetite + Global Martian Dust Analogue
fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

spectra = [
    (AEFE01_angle, AEFE01_intensity, AEFE01_peaks, "Hematite (AEFE01)", "red"),
    (AEFE02_angle, AEFE02_intensity, AEFE02_peaks, "Magnetite (AEFE02)", "black"),
    (AEMA01_angle, AEMA01_intensity, AEMA01_peaks, "Global Martian Dust Analogue (AEMA01)", "blue")
]

for ax, (angle, intensity, peaks, label, color) in zip(axes, spectra):

    ax.plot(angle, intensity, color=color, label=label)
    ax.scatter(*peaks, color="brown", marker="x")

    for wn, tr in zip(*peaks):
        ax.text(wn, tr, f"{wn:.1f}", fontsize=10)

    ax.set_ylabel("Normalized Intensity (counts)")
    ax.legend()
    ax.grid()

axes[-1].set_xlabel(f"2{chr(977)} (°)")

plt.suptitle("XRD Spectra Comparison")
plt.tight_layout()
plt.savefig("XRDPlots\\Stacked_Spectra.png", dpi=500)
plt.close()

# Hematite + RUFF Hematite
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

spectra = [
    (AEFE01_angle, AEFE01_intensity, AEFE01_peaks, "Hematite (AEFE01)", "red"),
    (AEFE01_RUFF_angle, AEFE01_RUFF_intensity, AEFE01_RUFF_peaks, " RUFF Hematite", "red"),
]

for ax, (angle, intensity, peaks, label, color) in zip(axes, spectra):

    ax.plot(angle, intensity, color=color, label=label)
    ax.scatter(*peaks, color="brown", marker="x")

    for wn, tr in zip(*peaks):
        ax.text(wn, tr, f"{wn:.1f}", fontsize=10)

    ax.set_ylabel("Normalized Intensity (counts)")
    ax.legend()
    ax.grid()

axes[-1].set_xlabel(f"2{chr(977)} (°)")

plt.suptitle("XRD Hematite Spectra Comparison")
plt.tight_layout()
plt.savefig("XRDPlots\\Stacked_Spectra_Hematite.png", dpi=500)
plt.close()

# Magnetite + RUFF Magnetite
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

spectra = [
    (AEFE02_angle, AEFE02_intensity, AEFE02_peaks, "Magnetite (AEFE02)", "black"),
    (AEFE02_RUFF_angle, AEFE02_RUFF_intensity, AEFE02_RUFF_peaks, " RUFF Magnetite", "black"),
]

for ax, (angle, intensity, peaks, label, color) in zip(axes, spectra):

    ax.plot(angle, intensity, color=color, label=label)
    ax.scatter(*peaks, color="brown", marker="x")

    for wn, tr in zip(*peaks):
        ax.text(wn, tr, f"{wn:.1f}", fontsize=10)

    ax.set_ylabel("Normalized Intensity (counts)")
    ax.legend()
    ax.grid()

axes[-1].set_xlabel(f"2{chr(977)} (°)")

plt.suptitle("XRD Magnetite Spectra Comparison")
plt.tight_layout()
plt.savefig("XRDPlots\\Stacked_Spectra_Magnetite.png", dpi=500)
plt.close()