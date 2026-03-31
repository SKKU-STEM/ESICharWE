import warnings
import matplotlib
import matplotlib.pyplot as plt

# Suppress known irrelevant warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="hyperspy")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# --- C K Analysis ---
CK_ENERGY_MIN = 270.0
CK_ENERGY_MAX = 330.0
CK_STRIDE_BINNING = 2
CK_NMF_MAX_ITER = 20000
CK_KMEANS_MAX_ITER = 1000

# --- O K Analysis ---
OK_ENERGY_MIN = 500.0
OK_ENERGY_MAX = 560.0
OK_STRIDE_BINNING = 2
OK_N_PCA = 4
OK_SAVGOL_WINDOW = 21
OK_SAVGOL_POLYORDER = 2
OK_BG_FIT_RANGE = (500, 528)
OK_KMEANS_CLUSTERS = 3
OK_KMEANS_MAX_ITER = 1000

# Vo calibration (linear mapping from A/B ratio to vacancy concentration)
# Vo = (A/B - 0.492) / 0.017, clipped at 0
VO_INTERCEPT = 0.492
VO_SLOPE = 0.017
