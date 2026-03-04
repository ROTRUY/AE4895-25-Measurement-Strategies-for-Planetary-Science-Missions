import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# -----------------------------
# Load spectrum
# -----------------------------
def load_spectrum(filename):
    data = np.genfromtxt(
        filename,
        delimiter=",",
        skip_header=2,
        encoding="utf-8"
    )

    wavenumber = data[:, 0]

    # Combine integer and decimal columns
    transmittance = data[:, 1] + data[:, 2] / 100.0

    return wavenumber, transmittance


# -----------------------------
# Peak detection (absorption bands)
# -----------------------------
def detect_peaks(wavenumber, transmittance, prominence=1.5):
    inverted = -transmittance
    peaks, properties = find_peaks(inverted, prominence=prominence)
    return wavenumber[peaks], transmittance[peaks]


# -----------------------------
# Load spectra
# -----------------------------
AEMA01 = load_spectrum("AEMA01_V2.csv")
AEMA02 = load_spectrum("AEMA02.csv")

references = {
    "Hematite": {
        "data": load_spectrum("AEFE01.csv"),
        "color": "red"
    },
    "Magnetite": {
        "data": load_spectrum("AEFE02.csv"),
        "color": "black"
    },
    "Olivine": {
        "data": load_spectrum("AEOL01_V2.csv"),
        "color": "green"
    },
    "Quartz": {
        "data": load_spectrum("AEQ01.csv"),
        "color": "pink"
    },
}


# -----------------------------
# Detect analogue peaks
# -----------------------------
analogue_peaks_1 = detect_peaks(*AEMA01)
analogue_peaks_2 = detect_peaks(*AEMA02)


# =============================
# 1️⃣ SUBPLOTS WITH PEAK LABELS
# =============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (ref_name, ref_info) in zip(axes, references.items()):

    wn_ref, tr_ref = ref_info["data"]
    ref_color = ref_info["color"]

    ref_peaks = detect_peaks(wn_ref, tr_ref)

    # Plot analogue samples
    ax.plot(*AEMA01, linestyle="--", color="blue", label="AEMA01")
    ax.plot(*AEMA02, linestyle="--", color="orange", label="AEMA02")

    # Plot reference sample
    ax.plot(wn_ref, tr_ref, linestyle="-", color=ref_color, label=ref_name)

    # Mark analogue peaks
    ax.scatter(*analogue_peaks_1, color="blue")
    ax.scatter(*analogue_peaks_2, color="orange")

    # Mark reference peaks
    ax.scatter(*ref_peaks, color=ref_color)

    # ---- Add peak labels ----
    for wn, tr in zip(*analogue_peaks_1):
        ax.text(wn, tr, f"{wn:.0f}", fontsize=8, color="blue")

    for wn, tr in zip(*analogue_peaks_2):
        ax.text(wn, tr, f"{wn:.0f}", fontsize=8, color="orange")

    for wn, tr in zip(*ref_peaks):
        ax.text(wn, tr, f"{wn:.0f}", fontsize=8, color=ref_color)

    # Formatting
    ax.set_title(f"Analogue vs {ref_name}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("FTIR_per_sample_peaks_comparison.png", dpi=300)
plt.show()
plt.close()


# =============================
# 2️⃣ FULL COMPARISON PLOT (NO PEAKS)
# =============================
plt.figure(figsize=(10, 6))

# Analogue samples (dashed)
plt.plot(*AEMA01, linestyle="--", color="blue", label="AEMA01")
plt.plot(*AEMA02, linestyle="--", color="orange", label="AEMA02")

# Reference samples (solid with requested colors)
for ref_name, ref_info in references.items():
    wn, tr = ref_info["data"]
    plt.plot(wn, tr, linestyle="-", color=ref_info["color"], label=ref_name)

plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Transmittance (%)")
plt.title("FTIR Spectra - Mars Analogues and Reference Minerals")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("FTIR_comparison.png", dpi=300)
plt.show()