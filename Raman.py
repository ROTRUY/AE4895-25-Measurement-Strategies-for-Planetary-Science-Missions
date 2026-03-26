import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from pybaselines import Baseline

def read_spectrum(filepath: str, min_wave:float=None, max_wave:float=None):
    data = np.loadtxt(filepath, comments='#')

    wave = data[:, 0]
    intensity = data[:, 1]

    if min_wave is not None or max_wave is not None:
        mask = np.ones_like(wave, dtype=bool)

        if min_wave is not None:
            mask &= wave >= min_wave
        if max_wave is not None:
            mask &= wave <= max_wave

        wave = wave[mask]
        intensity = intensity[mask]

    return wave, intensity

def read_spectrum_csv(filepath: str, min_wave:float=None, max_wave:float=None):
    data = np.loadtxt(filepath, delimiter=",")

    wave = data[:, 0]
    intensity = data[:, 1]

    if min_wave is not None or max_wave is not None:
        mask = np.ones_like(wave, dtype=bool)

        if min_wave is not None:
            mask &= wave >= min_wave
        if max_wave is not None:
            mask &= wave <= max_wave

        wave = wave[mask]
        intensity = intensity[mask]

    return wave, intensity

def detect_peaks(wave, intensity, prominence=200, distance=1):
    peaks, _ = find_peaks(intensity, prominence=prominence, distance=distance)

    return wave[peaks], intensity[peaks]

def baseline_correct(wave, intensity):
    baseline_fitter = Baseline(x_data=wave)
    baseline, params = baseline_fitter.asls(intensity, lam=1e7)
    corrected = intensity - baseline
    return corrected

def normalize(arr):
    return arr / np.max(arr)

### READ SPECTRA, SMOOTH AND NORMALIZE ###
# Analogues
AEMA01_wave, AEMA01_intensity = read_spectrum(r"RamanTxt\sample1.txt", 100, 1500)   # Martian dust analogue
AEMA01_intensity = baseline_correct(AEMA01_wave, AEMA01_intensity)                  # Baseline correction
AEMA01_intensity = savgol_filter(AEMA01_intensity, 11, 3)                           # Smooth the spectrum using Savitzky-Golay filter
AEMA01_intensity = normalize(AEMA01_intensity)                                      # Normalize intensity

AEMA02_wave, AEMA02_intensity = read_spectrum(r"RamanTxt\sample2.txt", 100, 1500)   # Jezero dust analogue
AEMA02_intensity = baseline_correct(AEMA02_wave, AEMA02_intensity)                  # Baseline correction
AEMA02_intensity = savgol_filter(AEMA02_intensity, 11, 3)                           # Smooth the spectrum using Savitzky-Golay filter
AEMA02_intensity = normalize(AEMA02_intensity)                                      # Normalize intensity

# References
AEFE01_wave, AEFE01_intensity = read_spectrum(r"RamanTxt\sample3v2.txt", 275, 500)  # Hematite reference
AEFE01_intensity = baseline_correct(AEFE01_wave, AEFE01_intensity)                  # Baseline correction
AEFE01_intensity = savgol_filter(AEFE01_intensity, 11, 3)                           # Smooth the spectrum using Savitzky-Golay filter
AEFE01_intensity = normalize(AEFE01_intensity)                                      # Normalize intensity

AEFE02_wave, AEFE02_intensity = read_spectrum(r"RamanTxt\sample4v2.txt", 500, 800)  # Magnetite reference
AEFE02_intensity = baseline_correct(AEFE02_wave, AEFE02_intensity)                  # Baseline correction
AEFE02_intensity = savgol_filter(AEFE02_intensity, 11, 3)                           # Smooth the spectrum using Savitzky-Golay filter
AEFE02_intensity = normalize(AEFE02_intensity)                                      # Normalize intensity

AEOL01_wave, AEOL01_intensity = read_spectrum(r"RamanTxt\sample5v2.txt", 800, 1000)  # Olivine reference
AEOL01_intensity = baseline_correct(AEOL01_wave, AEOL01_intensity)                   # Baseline correction
AEOL01_intensity = savgol_filter(AEOL01_intensity, 11, 3)                            # Smooth the spectrum using Savitzky-Golay filter
AEOL01_intensity = normalize(AEOL01_intensity)                                       # Normalize intensity

AEQ01_wave,  AEQ01_intensity  = read_spectrum(r"RamanTxt\sample6v2.txt", 180, 500)  # Quartz reference
AEQ01_intensity = baseline_correct(AEQ01_wave, AEQ01_intensity)                     # Baseline correction
AEQ01_intensity = savgol_filter(AEQ01_intensity, 11, 3)                             # Smooth the spectrum using Savitzky-Golay filter
AEQ01_intensity = normalize(AEQ01_intensity)                                        # Normalize intensity

# RUFF References
AEFE01_RUFF_wave, AEFE01_RUFF_intensity = read_spectrum_csv(r"RamanTxt\Hematite_RUFF.csv", 275, 500)  # Hematite RUFF reference
AEFE01_RUFF_intensity = baseline_correct(AEFE01_RUFF_wave, AEFE01_RUFF_intensity)                     # Baseline correction
AEFE01_RUFF_intensity = savgol_filter(AEFE01_RUFF_intensity, 11, 3)                                   # Smooth the spectrum using Savitzky-Golay filter
AEFE01_RUFF_intensity = normalize(AEFE01_RUFF_intensity)                                              # Normalize intensity

AEFE02_RUFF_wave, AEFE02_RUFF_intensity = read_spectrum_csv(r"RamanTxt\Magnetite_RUFF.csv", 500, 800)  # Magnetite RUFF reference
AEFE02_RUFF_intensity = baseline_correct(AEFE02_RUFF_wave, AEFE02_RUFF_intensity)                      # Baseline correction
AEFE02_RUFF_intensity = savgol_filter(AEFE02_RUFF_intensity, 11, 3)                                    # Smooth the spectrum using Savitzky-Golay filter
AEFE02_RUFF_intensity = normalize(AEFE02_RUFF_intensity)                                               # Normalize intensity

AEOL01_RUFF_wave, AEOL01_RUFF_intensity = read_spectrum_csv(r"RamanTxt\Olivine_RUFF.csv", 800, 1000)   # Olivine RUFF reference
AEOL01_RUFF_intensity = baseline_correct(AEOL01_RUFF_wave, AEOL01_RUFF_intensity)                      # Baseline correction
AEOL01_RUFF_intensity = savgol_filter(AEOL01_RUFF_intensity, 11, 3)                                    # Smooth the spectrum using Savitzky-Golay filter
AEOL01_RUFF_intensity = normalize(AEOL01_RUFF_intensity)                                               # Normalize intensity

AEQ01_RUFF_wave, AEQ01_RUFF_intensity = read_spectrum_csv(r"RamanTxt\Quartz_RUFF.csv", 180, 500)     # Quartz RUFF reference
AEQ01_RUFF_intensity = baseline_correct(AEQ01_RUFF_wave, AEQ01_RUFF_intensity)                       # Baseline correction
AEQ01_RUFF_intensity = savgol_filter(AEQ01_RUFF_intensity, 11, 3)                                    # Smooth the spectrum using Savitzky-Golay filter
AEQ01_RUFF_intensity = normalize(AEQ01_RUFF_intensity)                                               # Normalize intensity

### GET PEAKS ###
# Analogues
AEMA01_peaks = detect_peaks(AEMA01_wave, AEMA01_intensity, 0.1)
AEMA02_peaks = detect_peaks(AEMA02_wave, AEMA02_intensity, 0.1)

# References
AEFE01_peaks = detect_peaks(AEFE01_wave, AEFE01_intensity, 0.1)
AEFE02_peaks = detect_peaks(AEFE02_wave, AEFE02_intensity, 0.1)
AEOL01_peaks = detect_peaks(AEOL01_wave, AEOL01_intensity, 0.001)
AEQ01_peaks  = detect_peaks(AEQ01_wave,  AEQ01_intensity,  0.01)

# RUFF References
AEFE01_RUFF_peaks = detect_peaks(AEFE01_RUFF_wave, AEFE01_RUFF_intensity, 0.1)
AEFE02_RUFF_peaks = detect_peaks(AEFE02_RUFF_wave, AEFE02_RUFF_intensity, 0.1)
AEOL01_RUFF_peaks = detect_peaks(AEOL01_RUFF_wave, AEOL01_RUFF_intensity, 0.001)
AEQ01_RUFF_peaks = detect_peaks(AEQ01_RUFF_wave, AEQ01_RUFF_intensity, 0.01)

### DEFINE SPECTRA FOR PLOTTING ###
analogues = [
    ("AEMA01", AEMA01_wave, AEMA01_intensity, "Martian Dust Analogue (AEMA01)", "blue", AEMA01_peaks),
    ("AEMA02", AEMA02_wave, AEMA02_intensity, "Jezero Dust Analogue (AEMA02)",  "orange", AEMA02_peaks),
]

references = [
    ("AEFE01", AEFE01_wave, AEFE01_intensity, "Hematite (AEFE01)",  "red", AEFE01_peaks),
    ("AEFE02", AEFE02_wave, AEFE02_intensity, "Magnetite (AEFE02)", "black", AEFE02_peaks),
    ("AEOL01", AEOL01_wave, AEOL01_intensity, "Olivine (AEOL01)",   "green", AEOL01_peaks),
    ("AEQ01",  AEQ01_wave,  AEQ01_intensity,  "Quartz (AEQ01)",    "pink", AEQ01_peaks),
]

### PLOTTING ###
# New individual plots of analogues
# for item in analogues:
#     plt.figure(figsize=(14, 6))
#     plt.plot(item[1], item[2], label=item[3], color=item[4])
#     # Mark peaks
#     peakColour = "brown"
#     plt.scatter(*item[5], color=peakColour, marker='x')
#     for wn, tr in zip(*item[5]):
#         plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
#     plt.xlabel('Wave Number (cm⁻¹)')
#     plt.ylabel('Relative Intensity (counts)')
#     plt.title(f"Raman Spectrum of {item[3]} with peaks")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f"RamanPlots\\{item[0]}_WP_N.png", dpi=500)
#     plt.close()

# New individual plots of references
# for item in references:
#     plt.figure(figsize=(14, 6))
#     plt.plot(item[1], item[2], label=item[3], color=item[4])
#     # Mark peaks
#     peakColour = "brown"
#     plt.scatter(*item[5], color=peakColour, marker='x')
#     for wn, tr in zip(*item[5]):
#         plt.text(wn, tr, f"{wn:.1f}", fontsize=12, color=peakColour)
#     plt.xlabel('Wave Number (cm⁻¹)')
#     plt.ylabel('Relative Intensity (counts)')
#     plt.title(f"Raman Spectrum of {item[3]} with peaks")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f"RamanPlots\\{item[0]}_WP_N.png", dpi=500)
#     plt.close()

# Stacked Hematite + RUFF Hematite
# fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
# 
# spectra = [
#     (AEFE01_wave, AEFE01_intensity, AEFE01_peaks, "Hematite (AEFE01)", "red"),
#     (AEFE01_RUFF_wave, AEFE01_RUFF_intensity, AEFE01_RUFF_peaks, " RUFF Hematite", "red"),
# ]
# 
# for ax, (wave, intensity, peaks, label, color) in zip(axes, spectra):
# 
#     ax.plot(wave, intensity, color=color, label=label)
#     ax.scatter(*peaks, color="brown", marker="x")
# 
#     for wn, tr in zip(*peaks):
#         ax.text(wn, tr, f"{wn:.1f}", fontsize=10)
# 
#     ax.set_ylabel("Relative Intensity (counts)")
#     ax.legend()
#     ax.grid()
# 
# axes[-1].set_xlabel("Wave Number (cm⁻¹)")
# 
# plt.suptitle("Raman Hematite Spectra Comparison")
# plt.tight_layout()
# plt.savefig("RamanPlots\\Stacked_Spectra_Hematite.png", dpi=500)
# plt.close()

# Stacked Magnetite + RUFF Magnetite
# fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
# 
# spectra = [
#     (AEFE02_wave, AEFE02_intensity, AEFE02_peaks, "Magnetite (AEFE02)", "black"),
#     (AEFE02_RUFF_wave, AEFE02_RUFF_intensity, AEFE02_RUFF_peaks, " RUFF Magnetite", "black"),
# ]
# 
# for ax, (wave, intensity, peaks, label, color) in zip(axes, spectra):
# 
#     ax.plot(wave, intensity, color=color, label=label)
#     ax.scatter(*peaks, color="brown", marker="x")
# 
#     for wn, tr in zip(*peaks):
#         ax.text(wn, tr, f"{wn:.1f}", fontsize=10)
# 
#     ax.set_ylabel("Relative Intensity (counts)")
#     ax.legend()
#     ax.grid()
# 
# axes[-1].set_xlabel("Wave Number (cm⁻¹)")
# 
# plt.suptitle("Raman Magnetite Spectra Comparison")
# plt.tight_layout()
# plt.savefig("RamanPlots\\Stacked_Spectra_Magnetite.png", dpi=500)
# plt.close()

# Stacked Olivine + RUFF Olivine
# fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
# 
# spectra = [
#     (AEOL01_wave, AEOL01_intensity, AEOL01_peaks, "Olivine (AEOL01)", "green"),
#     (AEOL01_RUFF_wave, AEOL01_RUFF_intensity, AEOL01_RUFF_peaks, " RUFF Olivine", "green"),
# ]
# 
# for ax, (wave, intensity, peaks, label, color) in zip(axes, spectra):
# 
#     ax.plot(wave, intensity, color=color, label=label)
#     ax.scatter(*peaks, color="brown", marker="x")
# 
#     for wn, tr in zip(*peaks):
#         ax.text(wn, tr, f"{wn:.1f}", fontsize=10)
# 
#     ax.set_ylabel("Relative Intensity (counts)")
#     ax.legend()
#     ax.grid()
# 
# axes[-1].set_xlabel("Wave Number (cm⁻¹)")
# 
# plt.suptitle("Raman Olivine Spectra Comparison")
# plt.tight_layout()
# plt.savefig("RamanPlots\\Stacked_Spectra_Olivine.png", dpi=500)
# plt.close()

# Stacked Quartz + RUFF Quartz
# fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
# 
# spectra = [
#     (AEQ01_wave, AEQ01_intensity, AEQ01_peaks, "Quartz (AEQ01)", "blue"),
#     (AEQ01_RUFF_wave, AEQ01_RUFF_intensity, AEQ01_RUFF_peaks, " RUFF Quartz", "blue"),
# ]
# 
# for ax, (wave, intensity, peaks, label, color) in zip(axes, spectra):
# 
#     ax.plot(wave, intensity, color=color, label=label)
#     ax.scatter(*peaks, color="brown", marker="x")
# 
#     for wn, tr in zip(*peaks):
#         ax.text(wn, tr, f"{wn:.1f}", fontsize=10)
# 
#     ax.set_ylabel("Relative Intensity (counts)")
#     ax.legend()
#     ax.grid()
# 
# axes[-1].set_xlabel("Wave Number (cm⁻¹)")
# 
# plt.suptitle("Raman Quartz Spectra Comparison")
# plt.tight_layout()
# plt.savefig("RamanPlots\\Stacked_Spectra_Quartz.png", dpi=500)
# plt.close()

# Stacked Analogues vs References
# for analogue in analogues:
# 
#     fig, axes = plt.subplots(5, 1, figsize=(30, 15), sharex=True)
# 
#     spectra = [
#         analogue,
#         references[0],
#         references[1],
#         references[2],
#         references[3],
#     ]
# 
#     for ax, (name, wave, intensity, label, color, peaks) in zip(axes, spectra):
# 
#         ax.plot(wave, intensity, color=color, label=label)
#         ax.scatter(*peaks, color="brown", marker="x")
# 
#         for wn, tr in zip(*peaks):
#             ax.text(wn, tr, f"{wn:.1f}", fontsize=9)
# 
#         ax.set_ylabel("Relative Intensity (counts)")
#         ax.legend()
#         ax.grid()
# 
#     axes[-1].set_xlabel("Wave Number (cm⁻¹)")
# 
#     plt.suptitle(f"Raman Analogue vs References Comparison ({analogue[3]})")
#     plt.tight_layout()
# 
#     plt.savefig(f"RamanPlots\\Stacked_{analogue[0]}_vs_References.png", dpi=500)
#     plt.close()

# Analogue and References in same plots
# for analogue in analogues:
# 
#     plt.figure(figsize=(14,8))
# 
#     offset = 0
#     offset_step = 0.6
# 
#     spectra = [analogue] + references
# 
#     for name, wave, intensity, label, color, peaks in spectra:
# 
#         plt.plot(wave, intensity + offset, label=label, color=color)
# 
#         # plot peaks
#         peak_wave, peak_int = peaks
#         plt.scatter(peak_wave, peak_int + offset, color="brown", marker="x", s=40)
# 
#         offset += offset_step
# 
#     plt.xlabel("Wave Number (cm⁻¹)")
#     plt.ylabel("Relative Intensity (offset)")
#     plt.title(f"Raman Comparison: {analogue[3]} vs References")
#     plt.legend()
#     plt.grid()
# 
#     plt.tight_layout()
#     plt.savefig(f"RamanPlots\\Overlay_{analogue[0]}_vs_References.png", dpi=500)
#     plt.close()

# Plot all Raman spectra stacked together
fig = plt.figure(figsize=(8.5, 7))

offset = 0
offset_step = 0.8

all_spectra = analogues + references

for name, wave, intensity, label, color, peaks in all_spectra:
    plt.plot(wave, intensity + offset, label=label, color=color, linewidth=2.5)
    
    # Plot peaks with offset
    peak_wave, peak_int = peaks
    plt.scatter(peak_wave, peak_int + offset, color="brown", marker="x", s=60, linewidth=2, zorder=5)
    
    # Add peak labels with better positioning
    for wn, inten in zip(peak_wave, peak_int):
        plt.text(wn, inten + offset + 0.05, f"{wn:.0f}", fontsize=8, color="brown", 
                ha='center', va='bottom', fontweight='bold')
    
    offset += offset_step

plt.xlabel("Wavenumber (cm⁻¹)", fontsize=13, fontweight='bold')
plt.ylabel("Relative Intensity (a.u.)", fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()

plt.savefig("RamanPlots\\Stacked_All_Spectra.png", dpi=500, bbox_inches='tight')
plt.close()

# Plot all Raman spectra stacked together with RUFF Hematite and Magnetite
fig = plt.figure(figsize=(8.5, 7))

offset = 0
offset_step = 0.8

# Use RUFF versions for hematite and magnetite instead of own data
ruff_analogues_and_references = [
    ("AEMA01", AEMA01_wave, AEMA01_intensity, "Martian Dust Analogue (AEMA01)", "blue", AEMA01_peaks),
    ("AEMA02", AEMA02_wave, AEMA02_intensity, "Jezero Dust Analogue (AEMA02)",  "orange", AEMA02_peaks),
    ("AEFE01_RUFF", AEFE01_RUFF_wave, AEFE01_RUFF_intensity, "Hematite (RUFF Reference)", "red", AEFE01_RUFF_peaks),
    ("AEFE02_RUFF", AEFE02_RUFF_wave, AEFE02_RUFF_intensity, "Magnetite (RUFF Reference)", "black", AEFE02_RUFF_peaks),
    ("AEOL01", AEOL01_wave, AEOL01_intensity, "Olivine (AEOL01)",   "green", AEOL01_peaks),
    ("AEQ01",  AEQ01_wave,  AEQ01_intensity,  "Quartz (AEQ01)",    "pink", AEQ01_peaks),
]

for name, wave, intensity, label, color, peaks in ruff_analogues_and_references:
    plt.plot(wave, intensity + offset, label=label, color=color, linewidth=2.5)
    
    # Plot peaks with offset
    peak_wave, peak_int = peaks
    plt.scatter(peak_wave, peak_int + offset, color="brown", marker="x", s=60, linewidth=2, zorder=5)
    
    # Add peak labels with better positioning
    for wn, inten in zip(peak_wave, peak_int):
        plt.text(wn, inten + offset + 0.05, f"{wn:.0f}", fontsize=8, color="brown", 
                ha='center', va='bottom', fontweight='bold')
    
    offset += offset_step

plt.xlabel("Wavenumber (cm⁻¹)", fontsize=13, fontweight='bold')
plt.ylabel("Relative Intensity (a.u.)", fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black')
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()

plt.savefig("RamanPlots\\Stacked_All_Spectra_RUFF.png", dpi=500, bbox_inches='tight')
plt.close()

# Plot just the two analogues stacked together
fig = plt.figure(figsize=(12, 6))

offset = 0
offset_step = 0.8

analogues_only = [
    ("AEMA01", AEMA01_wave, AEMA01_intensity, "Martian Dust Analogue (AEMA01)", "blue", AEMA01_peaks),
    ("AEMA02", AEMA02_wave, AEMA02_intensity, "Jezero Dust Analogue (AEMA02)", "orange", AEMA02_peaks),
]

for name, wave, intensity, label, color, peaks in analogues_only:
    plt.plot(wave, intensity + offset, label=label, color=color, linewidth=2.5)

    # peaks + labels for analogue spectra
    peak_wave, peak_int = peaks
    plt.scatter(peak_wave, peak_int + offset, color="brown", marker="x", s=60, linewidth=2, zorder=5)
    for wn, inten in zip(peak_wave, peak_int):
        plt.text(wn, inten + offset + 0.05, f"{wn:.0f}", fontsize=8, color="brown", ha='center', va='bottom')

    offset += offset_step

plt.xlabel("Wavenumber (cm⁻¹)", fontsize=13, fontweight='bold')
plt.ylabel("Relative Intensity (a.u.)", fontsize=13, fontweight='bold')
plt.title("Stacked Analogues Only (AEMA01 & AEMA02)", fontsize=14, fontweight='bold', pad=10)
plt.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()

plt.savefig("RamanPlots\\Stacked_Analogues_Only.png", dpi=500, bbox_inches='tight')
plt.close()