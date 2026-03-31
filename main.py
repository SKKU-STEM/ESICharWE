import warnings
import config  


def _prompt_str(message):
    while True:
        val = input(message).strip()
        if val:
            return val
        print("  Input cannot be empty.")


def _prompt_int(message, min_val, max_val, default=None):
    hint = f" [{min_val}-{max_val}]"
    if default is not None:
        hint += f" (default: {default})"
    hint += ": "
    while True:
        raw = input(message + hint).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input. Enter an integer.")


def _prompt_float(message, default=None):
    hint = f" (default: {default}): " if default is not None else ": "
    while True:
        raw = input(message + hint).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Invalid input. Enter a number.")


def _prompt_yes_no(message, default=True):
    hint = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(message + hint).strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n.")

def main():
    print("=" * 50)
    print("  STEM-EELS Mapping Program")
    print("=" * 50)
    print("  1. C K edge analysis (NMF + K-means)")
    print("  2. O K edge analysis (A/B ratio + Vo map)")
    print("  3. Combined map overlay")
    print("  q. Quit")
    print("=" * 50)

    choice = _prompt_str("Select menu").lower()

    if choice == "1":
        _run_ck()
    elif choice == "2":
        _run_ok()
    elif choice == "3":
        _run_map()
    elif choice == "q":
        print("Exiting.")
    else:
        print("Invalid selection.")


def _load_eels():
    import hyperspy.api as hs

    filepath = _prompt_str("Put file path of raw EELS data (.dm3 / .dm4)")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*signal_type='EELS' not understood.*"
        )
        sig = hs.load(filepath, signal_type="EELS")

    print(f"\nLoaded: {filepath}")
    print(sig)
    print(sig.axes_manager)

    if sig.axes_manager[-1].offset <= 0:
        print("  Energy offset <= 0 detected.")
        offset = _prompt_float("  Enter energy-axis offset (eV)", default=220.0)
        sig.axes_manager[-1].offset = offset
        print(f"  Energy offset set to {offset} eV.")

    return sig, filepath


def _run_ck():
    from analysis_ck import perform_ck_analysis
    import matplotlib.pyplot as plt

    print("\n--- C K Edge Analysis ---")
    sig, filepath = _load_eels()

    sig.plot()
    plt.show()

    n = _prompt_int("Number of NMF components", min_val=2, max_val=10, default=3)
    save = _prompt_yes_no("Save results to disk", default=True)

    perform_ck_analysis(sig, filepath, n_components=n, save=save)


def _run_ok():
    from analysis_ok import perform_ok_analysis

    print("\n--- O K Edge Analysis ---")
    sig, filepath = _load_eels()
    save = _prompt_yes_no("Save results to disk", default=True)

    perform_ok_analysis(sig, filepath, save=save)


def _run_map():
    import tifffile as tiff
    from visualization import combine_map

    print("\n--- Combined Map Overlay ---")
    adf        = tiff.imread(_prompt_str("ADF image path (.tif)"))
    ck_ionomer = tiff.imread(_prompt_str("C K ionomer cluster mask path (.tif)"))
    ck_irox    = tiff.imread(_prompt_str("C K IrOx cluster mask path (.tif)"))
    ok_vacancy = tiff.imread(_prompt_str("O K vacancy map path (.tif)"))

    combine_map(adf, ck_ionomer, ck_irox, ok_vacancy)


if __name__ == "__main__":
    main()
