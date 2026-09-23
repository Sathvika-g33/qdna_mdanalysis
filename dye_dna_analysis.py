from pathlib import Path
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


# ---- PLOTTING FUNCTIONS ----
def plot_distance_vs_time(
    time, distance, output_dir=Path("outputs"), filename="distance_vs_time.png"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(time, distance)
    plt.xlabel("Time (ps)")
    plt.ylabel("Min Distance to DNA (Å)")
    plt.title("Dye–DNA Minimum Distance vs Time")
    plt.savefig(output_dir / filename)
    plt.close()


def plot_angle_vs_time(
    time, angle, output_dir=Path("outputs"), filename="angle_vs_time.png"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(time, angle)
    plt.xlabel("Time (ps)")
    plt.ylabel("Angle to DNA Axis (°)")
    plt.title("Dye–DNA Angle vs Time")
    plt.savefig(output_dir / filename)
    plt.close()


def plot_angle_vs_distance_heatmap(
    distance,
    angle,
    output_dir=Path("outputs"),
    filename="angle_vs_distance_heatmap.png",
    bins=100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist2d(distance, angle, bins=bins, cmap="viridis")
    plt.colorbar(label="Counts")
    plt.xlabel("Min Distance to DNA (Å)")
    plt.ylabel("Angle to DNA Axis (°)")
    plt.title("Dye–DNA Angle vs Distance Heatmap")
    plt.savefig(output_dir / filename)
    plt.close()


# ---- DATA SAVING FUNCTIONS ----
def save_analysis_csv(
    time,
    distance,
    angle,
    output_dir=Path("outputs"),
    filename="analysis_results.csv",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    header = "time,distance,angle\n"
    with open(csv_path, "w") as csv_file:
        csv_file.write(header)
        for t, d, a in zip(time, distance, angle):
            csv_file.write(f"{t},{d},{a}\n")
    return csv_path


def save_analysis_log(
    time,
    distance,
    angle,
    output_dir=Path("outputs"),
    filename="analysis_results.log",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    with open(log_path, "w") as log_file:
        log_file.write("Analysis results\n")
        log_file.write("time\tdistance\tangle\n")
        for t, d, a in zip(time, distance, angle):
            log_file.write(f"{t:.3f}\t{d:.3f}\t{a:.2f}\n")
    return log_path


# ==============================================================================
# ---- CONFIGURATION / PARAMETERS (EDIT THESE VALUES FOR YOUR SYSTEM) ----
# ==============================================================================

# 1. File Paths
TOPOLOGY = "md_pdc_correct_cluster_920_start.pdb"     
TRAJECTORY = "md_pdc_correct_cluster_920.xtc"   

# 2. Dye Selection Parameters
DYE_RESNAME = "CY5"                  
DYE_RESNUM = 1                       # Residue number of the dye molecule

# 3. Dye End Atom Names (defines long molecular vector along dye conjugated core)
DYE_START_ATOMS = "name C16 C17 C12 C13 C14 C15"  # End 1 atom names
DYE_END_ATOMS = "name C4 C3 C2 C1 C5 C6"         # End 2 atom names

# 4. Nucleic Acid Selection Parameters
DNA_SELECTION = "nucleic"            # MDAnalysis selection query for DNA
DNA_BACKBONE_ATOM = "name P"         # Backbone reference atom name

# ==============================================================================

# ---- LOAD FILE ----
u = mda.Universe(TOPOLOGY, TRAJECTORY)

# ---- SELECTIONS ----
dye_selection_str = f"resname {DYE_RESNAME} and resnum {DYE_RESNUM}"
dye = u.select_atoms(dye_selection_str, updating=True)
dye_start = dye.select_atoms(DYE_START_ATOMS, updating=True)
dye_end = dye.select_atoms(DYE_END_ATOMS, updating=True)

dna = u.select_atoms(DNA_SELECTION, updating=True)

print("time distance angle")
alltime = []
distance = []
allangles = []

# ---- MAIN TRAJECTORY LOOP ----
for ts in u.trajectory:
    time = ts.time
    alltime.append(time)

    # 1. Minimum Distance from Dye COG to DNA using Python-level NumPy vector math
    dye_cog = dye.center_of_geometry()
    min_dist = np.min(np.linalg.norm(dna.positions - dye_cog, axis=1))
    distance.append(min_dist)

    # 2. Dye Vector (End 1 -> End 2)
    v_dye = dye_end.center_of_geometry() - dye_start.center_of_geometry()
    v_dye = v_dye / np.linalg.norm(v_dye)

    # 3. DNA Axis Vector (Phosphorus-to-Phosphorus or Terminal Atoms)
    dna_p = dna.select_atoms(DNA_BACKBONE_ATOM)
    if len(dna_p) >= 2:
        v_dna = dna_p[-1].position - dna_p[0].position
    else:
        v_dna = dna.positions[-1] - dna.positions[0]
    v_dna = v_dna / np.linalg.norm(v_dna)

    # 4. Acute Angle Calculation (0° to 90°)
    cos_theta = np.clip(np.dot(v_dye, v_dna), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    if angle > 90.0:
        angle = 180.0 - angle

    allangles.append(angle)

    # Frame-by-frame console output
    print(f"{time:.3f} {min_dist:.3f} {angle:.2f}")

print("Analysis Done.")

# ----- CONVERT TO NUMPY ARRAYS -------
runtime = np.array(alltime)
bonddistances = np.array(distance)
bondangles = np.array(allangles)

# ---- PLOTTING THE GRAPHS -----
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

plot_distance_vs_time(runtime, bonddistances, output_dir=output_dir)
plot_angle_vs_time(runtime, bondangles, output_dir=output_dir)
plot_angle_vs_distance_heatmap(
    bonddistances, bondangles, output_dir=output_dir
)

# ---- SAVE ANALYSIS DATA -----
save_analysis_csv(runtime, bonddistances, bondangles, output_dir=output_dir)
save_analysis_log(runtime, bonddistances, bondangles, output_dir=output_dir)

print("Analysis and plotting complete.")
