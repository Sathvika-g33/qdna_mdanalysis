import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- PLOTTING FUNCTIONS ----
def plot_distance_vs_time(time, distance, output_dir=Path("outputs"), filename="distance_vs_time.png"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(time, distance)
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Distance vs Time")
    plt.savefig(output_dir / filename)
    plt.close()


def plot_angle_vs_time(time, angle, output_dir=Path("outputs"), filename="angle_vs_time.png"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(time, angle)
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.title("Angle vs Time")
    plt.savefig(output_dir / filename)
    plt.close()


def plot_angle_vs_distance_heatmap(distance, angle, output_dir=Path("outputs"), filename="angle_vs_distance_heatmap.png", bins=100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist2d(distance, angle, bins=bins)
    plt.xlabel("Distance")
    plt.ylabel("Angle")
    plt.title("Angle vs Distance Heatmap")
    plt.savefig(output_dir / filename)
    plt.close()

# ---- DATA SAVING FUNCTIONS ----
def save_analysis_csv(time, distance, angle, output_dir=Path("outputs"), filename="analysis_results.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    header = "time,distance,angle\n"
    with open(csv_path, "w") as csv_file:
        csv_file.write(header)
        for t, d, a in zip(time, distance, angle):
            csv_file.write(f"{t},{d},{a}\n")
    return csv_path


def save_analysis_log(time, distance, angle, output_dir=Path("outputs"), filename="analysis_results.log"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    with open(log_path, "w") as log_file:
        log_file.write("Analysis results\n")
        log_file.write("time\tdistance\tangle\n")
        for t, d, a in zip(time, distance, angle):
            log_file.write(f"{t:.3f}\t{d:.3f}\t{a:.2f}\n")
    return log_path

# ---- LOAD FILE ----
u = mda.Universe("")

# ---- SELECTIONS ----
dye1 = u.select_atoms("resname CY5 and resnum 1", updating = True)
dye2 = u.select_atoms("resname CY5 and resnum 2", updating = True)
dye1_start = dye1.select_atoms("name C16 C17 C12 C13 C14 C15", updating = True)
dye1_end = dye1.select_atoms("name C4 C3 C2 C1 C5 C6",updating = True)
dye2_start = dye2.select_atoms("name C16 C17 C12 C13 C14 C15", updating = True)
dye2_end = dye2.select_atoms("name C4 C3 C2 C1 C5 C6",updating = True)

print("time distance angle")
alltime = []
distance = []
allangles =[]

# ---- LOOP ----
for ts in u.trajectory:

    time = ts.time
    alltime.append(time)

    # distance (COG)
    cog1 = dye1.center_of_geometry()
    cog2 = dye2.center_of_geometry()
    dist = np.linalg.norm(cog1 - cog2)
    distance.append(dist)

    # angle
    v1 = dye1_end.center_of_geometry() - dye1_start.center_of_geometry()
    v2 = dye2_end.center_of_geometry() - dye2_start.center_of_geometry()

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2)
    angle = np.degrees(np.arccos(cos_theta))
    allangles.append(angle)

    print(f"{time:.3f} {dist:.3f} {angle:.2f}")

print("Analysis Done.")
#----- CONVERT TO NUMPY ARRAYS -------
runtime = np.array(alltime)
bonddistances = np.array(distance)
bondangles = np.array(allangles)

#----PLOTTING THE GRAPHS -----
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

#--- Dist VS Time -----
plot_distance_vs_time(runtime, bonddistances, output_dir=output_dir)

#--- Angle VS Time -----
plot_angle_vs_time(runtime, bondangles, output_dir=output_dir)

#--- 2D heat map Angle VS Distance -----
plot_angle_vs_distance_heatmap(bonddistances, bondangles, output_dir=output_dir)

#--- Save analysis data -----
save_analysis_csv(runtime, bonddistances, bondangles, output_dir=output_dir)
save_analysis_log(runtime, bonddistances, bondangles, output_dir=output_dir)

print("Analysis and plotting complete.")