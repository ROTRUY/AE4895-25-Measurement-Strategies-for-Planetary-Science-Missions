import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def read_spectrum(filepath: str):
    """
    Reads Raman spectroscopy data from a text file.
    
    :param filepath: Path to the text file containing the spectrum data.
    :type filepath: str
    :return: Tuple of wave and intensity arrays.
    """
    data = np.loadtxt(filepath, comments='#')
    
    wave = data[:, 0]
    intensity = data[:, 1]
    
    return wave, intensity

def detect_peaks(angle, intensity, prominence=200, distance=1):
    peaks, _ = find_peaks(intensity, prominence=prominence, distance=distance)

    return angle[peaks], intensity[peaks]

AEMA01_wave, AEMA01_intensity = read_spectrum(r"RamanTxt\sample1.txt")    # Martian dust analogue
AEMA02_wave, AEMA02_intensity = read_spectrum(r"RamanTxt\sample2.txt")    # Jezero dust analogue
AEFE01_wave, AEFE01_intensity = read_spectrum(r"RamanTxt\sample3v2.txt")  # Hematite reference
AEFE02_wave, AEFE02_intensity = read_spectrum(r"RamanTxt\sample4v2.txt")  # Magnetite reference
AEOL01_wave, AEOL01_intensity = read_spectrum(r"RamanTxt\sample5v2.txt")  # Olivine reference
AEQ01_wave,  AEQ01_intensity  = read_spectrum(r"RamanTxt\sample6v2.txt")  # Quartz reference

# Get peaks
AEMA01_peaks = detect_peaks(AEMA01_wave, AEMA01_intensity, 450, 1)
AEMA02_peaks = detect_peaks(AEMA02_wave, AEMA02_intensity, 1300, 1)
AEFE01_peaks = detect_peaks(AEFE01_wave, AEFE01_intensity, 2000, 1)
AEFE02_peaks = detect_peaks(AEFE02_wave, AEFE02_intensity, 2000, 1)
AEOL01_peaks = detect_peaks(AEOL01_wave, AEOL01_intensity, 600, 1)
AEQ01_peaks  = detect_peaks(AEQ01_wave,  AEQ01_intensity,  500, 1)

def plot_spectra(x, y, peaks, title: str, labels=None, colours=None, linestyles=None, show: bool = True, save_path: str = None):
    """
    Plots one or multiple Raman spectra.

    :param x: List of wave arrays (or a single wave array).
    :type x: list of np.ndarray or np.ndarray
    :param y: List of intensity arrays (or a single intensity array).
    :type y: list of np.ndarray or np.ndarray
    :param title: Title of the plot.
    :type title: str
    :param labels: Optional list of labels for the legend.
    :type labels: list of str or str or None
    :param colors: Optional list of colors for the spectra.
    :type colors: list of str or str or None
    :param linestyles: Optional list of line styles for the spectra.
    :type linestyles: list of str or str or None
    :param show: Whether to display the plot.
    :type show: bool
    :param save_path: Optional path to save the plot.
    :type save_path: str or None
    """

    # Normalize x and y to list-of-arrays
    if isinstance(x, (np.ndarray, list, tuple)) and not isinstance(x[0], (np.ndarray, list, tuple)):
        x = [x]
        y = [y]

    n = len(x)

    if len(y) != n:
        raise ValueError("x and y must contain the same number of spectra.")
    
    # Normalize peaks to list-of-tuples
    if isinstance(peaks, tuple):
        peaks = [peaks]

    # Helper to normalize optional inputs
    def normalize(param, name):
        if param is None:
            return [None] * n
        if isinstance(param, str):
            return [param]
        if len(param) != n:
            raise ValueError(f"{name} must match number of spectra.")
        return param

    labels = normalize(labels, "labels")
    colours = normalize(colours, "colours")
    linestyles = normalize(linestyles, "linestyles")

    # Plotting
    plt.figure(figsize=(10, 6))

    for xi, yi, pi, label, colour, ls in zip(x, y, peaks, labels, colours, linestyles):
        plt.plot(xi, yi, label=label, color=colour, linestyle=ls)

        peak_x, peak_y = pi

        plt.scatter(peak_x, peak_y,
                    color="black",
                    marker='x',
                    zorder=3)

        # Optional peak labels
        for px, py in zip(peak_x, peak_y):
            plt.text(px, py,
                    f"{px:.0f}",
                    fontsize=9,
                    color="black",
                    ha="center",
                    va="bottom")

    plt.xlabel('Wave Number (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(title)

    if any(label is not None for label in labels):
        plt.legend()

    plt.grid()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    if show:
        plt.show()

# Define the spectra data and metadata for plotting
analogues = [
    ("AEMA01", AEMA01_wave, AEMA01_intensity, "Martian Dust Analogue", "blue", AEMA01_peaks),
    ("AEMA02", AEMA02_wave, AEMA02_intensity, "Jezero Dust Analogue",  "orange", AEMA02_peaks),
]

references = [
    ("AEFE01", AEFE01_wave, AEFE01_intensity, "Hematite",  "red", AEFE01_peaks),
    ("AEFE02", AEFE02_wave, AEFE02_intensity, "Magnetite", "black", AEFE02_peaks),
    ("AEOL01", AEOL01_wave, AEOL01_intensity, "Olivine",   "green", AEOL01_peaks),
    ("AEQ01",  AEQ01_wave,  AEQ01_intensity,  "Quartz",    "pink", AEQ01_peaks),
]


# Individual spectrum plots
plot_spectra(AEMA01_wave, AEMA01_intensity, AEMA01_peaks, title="Raman Spectrum of AEMA01 (Martian Dust Analogue) with peaks", labels="AEMA01 (Martian Dust Analogue)", colours="blue", show=False, save_path=r"RamanPlots\AEMA01_WP.png")
plot_spectra(AEMA02_wave, AEMA02_intensity, AEMA02_peaks, title="Raman Spectrum of AEMA02 (Jezero Dust Analogue) with peaks", labels="AEMA02 (Jezero Dust Analogue)", colours="orange", show=False, save_path=r"RamanPlots\AEMA02_WP.png")
plot_spectra(AEFE01_wave, AEFE01_intensity, AEFE01_peaks, title="Raman Spectrum of AEFE01 (Hematite) with peaks", labels="AEFE01 (Hematite)", colours="red", show=False, save_path=r"RamanPlots\AEFE01_WP.png")
plot_spectra(AEFE02_wave, AEFE02_intensity, AEFE02_peaks, title="Raman Spectrum of AEFE02 (Magnetite) with peaks", labels="AEFE02 (Magnetite)", colours="black", show=False, save_path=r"RamanPlots\AEFE02_WP.png")
plot_spectra(AEOL01_wave, AEOL01_intensity, AEOL01_peaks, title="Raman Spectrum of AEOL01 (Olivine) with peaks", labels="AEOL01 (Olivine)", colours="green", show=False, save_path=r"RamanPlots\AEOL01_WP.png")
plot_spectra(AEQ01_wave, AEQ01_intensity, AEQ01_peaks, title="Raman Spectrum of AEQ01 (Quartz) with peaks", labels="AEQ01 (Quartz)", colours="pink", show=False, save_path=r"RamanPlots\AEQ01_WP.png")
"""
# Analogue vs reference comparison plots
for a_code, a_wave, a_int, a_name, a_colour in analogues:
    for r_code, r_wave, r_int, r_name, r_colour in references:

        plot_spectra(
            [a_wave, r_wave],
            [a_int, r_int],
            title=f"{a_code} vs {r_name}",
            labels=[f"{a_name} ({a_code})", f"{r_name} ({r_code})"],
            colours=[a_colour, r_colour],
            save_path=rf"RamanPlots\{a_code}_vs_{r_name}.png"
        )
"""
# Analogues together with all references
for a_code, a_wave, a_int, a_name, a_colour, a_peaks in analogues:

    x_data = [a_wave]
    y_data = [a_int]
    labels = [f"{a_name} ({a_code})"]
    colours = [a_colour]
    peaks = [a_peaks]

    for r_code, r_wave, r_int, r_name, r_colour, r_peaks in references:
        x_data.append(r_wave)
        y_data.append(r_int)
        labels.append(f"{r_name} ({r_code})")
        colours.append(r_colour)
        peaks.append(r_peaks)

    plot_spectra(
        x_data,
        y_data,
        peaks,
        title=f"{a_code} vs Reference Minerals with peaks",
        labels=labels,
        colours=colours,
        show=False,
        save_path=rf"RamanPlots\{a_code}_vs_All_References_WP.png"
    )
