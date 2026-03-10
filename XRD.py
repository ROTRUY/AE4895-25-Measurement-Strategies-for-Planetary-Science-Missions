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

AEMA01_angle, AEMA01_intensity = read_spectrum(r"RamanTxt\sample1.txt")    # Martian dust analogue
AEFE01_angle, AEFE01_intensity = read_spectrum(r"RamanTxt\sample3v2.txt")  # Hematite reference
AEFE02_angle, AEFE02_intensity = read_spectrum(r"RamanTxt\sample4v2.txt")  # Magnetite reference

# Define the spectra data and metadata for plotting
to_plot = [
    ("AEMA01", AEMA01_angle, AEMA01_intensity, "Global Martian Dust Analogue", "blue"),
    ("AEFE01", AEFE01_angle, AEFE01_intensity, "Hematite",  "red"),
    ("AEFE02", AEFE02_angle, AEFE02_intensity, "Magnetite", "black")
]


plt.plot(to_plot[0][1], to_plot[0][2], label=to_plot[0][3], color=to_plot[0][4])
plt.plot(to_plot[1][1], to_plot[1][2], label=to_plot[1][3], color=to_plot[1][4])
plt.plot(to_plot[2][1], to_plot[2][2], label=to_plot[2][3], color=to_plot[2][4])
plt.xlabel('Angle (°)')
plt.ylabel('Intensity')
plt.title(f"XRD Spectrum of Global Martian Dust Analogue and References")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"XRDPlots\FullXRD.png")