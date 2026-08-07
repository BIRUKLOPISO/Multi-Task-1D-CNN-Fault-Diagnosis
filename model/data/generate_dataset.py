import numpy as np
import pandapower as pp
import pandapower.networks as nw
from tqdm import tqdm
import os
import warnings

# Suppress the numba warnings
warnings.filterwarnings("ignore")
pp.pandapower.__version__   # just to silence some internal logs

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SAVE_DIR = "dataset_real"
os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLES_PER_WINDOW = 333
NUM_SECTIONS = 12
NUM_CHANNELS = 18

FAULT_TYPES = list(range(11))          # 0–10
FAULT_RESISTANCES = [0.01, 1.0, 5.0, 10.0]
LOAD_LEVELS = [0.8, 1.0]
LOCATIONS = [0.3, 0.5, 0.7]

# -------------------------------------------------
# Create network
# -------------------------------------------------
def create_network():
    net = nw.case_ieee30()
    return net

# -------------------------------------------------
# Generate one sample
# -------------------------------------------------
def generate_sample(net, fault_type, section_idx, load_scale):
    if len(net.load) > 0:
        net.load.p_mw = net.load.p_mw * load_scale
        net.load.q_mvar = net.load.q_mvar * load_scale

    try:
        pp.runpp(net, numba=False)          # important: disable numba
    except:
        pass

    vm = net.res_bus.vm_pu.values
    va = np.deg2rad(net.res_bus.va_degree.values)

    t = np.linspace(0, 2/60, SAMPLES_PER_WINDOW)
    data = np.zeros((NUM_CHANNELS, SAMPLES_PER_WINDOW), dtype=np.float32)

    for i in range(3):                          # 3 measurement locations
        bus = i % len(vm)
        for ph in range(3):
            phase = ph * 2 * np.pi / 3
            # Voltage
            signal_v = vm[bus] * np.sin(2 * np.pi * 60 * t + va[bus] + phase)
            # Current
            signal_i = 0.4 * vm[bus] * np.sin(2 * np.pi * 60 * t + va[bus] + phase + np.pi/6)

            if fault_type != 10:
                start = int(0.25 * SAMPLES_PER_WINDOW)
                decay = np.exp(-7 * t[start:])
                fault_v = 0.22 * decay * np.sin(2 * np.pi * 120 * t[start:] + phase)
                fault_i = 0.35 * decay * np.sin(2 * np.pi * 100 * t[start:])
                fault_v *= (1 + 0.12 * fault_type + 0.07 * section_idx)
                fault_i *= (1 + 0.10 * fault_type + 0.09 * section_idx)
                signal_v[start:] += fault_v
                signal_i[start:] += fault_i

            data[i*6 + ph] = signal_v
            data[i*6 + 3 + ph] = signal_i

    data += np.random.normal(0, 0.007, data.shape)
    return data

# -------------------------------------------------
# Main
# -------------------------------------------------
def generate_real_dataset():
    net = create_network()
    X_list, y_type_list, y_section_list = [], [], []

    total = len(FAULT_TYPES) * NUM_SECTIONS * len(LOCATIONS) * len(FAULT_RESISTANCES) * len(LOAD_LEVELS)
    print(f"Total samples to generate: {total}")

    with tqdm(total=total) as pbar:
        for ftype in FAULT_TYPES:
            for section in range(NUM_SECTIONS):
                for loc in LOCATIONS:
                    for res in FAULT_RESISTANCES:
                        for load in LOAD_LEVELS:
                            wave = generate_sample(net, ftype, section, load)
                            X_list.append(wave)
                            y_type_list.append(ftype)
                            y_section_list.append(section)
                            pbar.update(1)

    X = np.stack(X_list)
    y_type = np.array(y_type_list)
    y_section = np.array(y_section_list)

    print("\nDataset shapes:")
    print("X:", X.shape)
    print("y_type:", y_type.shape)
    print("y_section:", y_section.shape)

    np.save(os.path.join(SAVE_DIR, "X.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_type.npy"), y_type)
    np.save(os.path.join(SAVE_DIR, "y_section.npy"), y_section)
    print(f"\nSaved in folder: {SAVE_DIR}")

if __name__ == "__main__":
    generate_real_dataset()
