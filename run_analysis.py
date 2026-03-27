import MDAnalysis as mda
import numpy as np
import matplolib.pyplot as plt
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
#--- Dist VS Time -----
plt.figure()
plt.plot(runtime, bonddistances)
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Distance vs Time")
plt.savefig("distance_vs_time.png")
plt.close()

#--- Angle VS Time -----
plt.figure()
plt.plot(runtime, bondangles)
plt.xlabel("Time")
plt.ylabel("Angle")
plt.title("Angle vs Time")
plt.savefig("angle_vs_time.png")
plt.close()

#--- 2D heat map Angle VS Distance -----
plt.figure()
plt.hist2d(bonddistances, bondangles, bins = 100)
plt.xlabel("Distance")
plt.ylabel("Angle")
plt.title("Angle vs Distance Heatmap")
plt.savefig("angle_vs_distance_heatmap.png")
plt.close()

Print("Analysis and plotting complete.")