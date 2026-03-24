import MDAnalysis as mda
import numpy as np

# ---- LOAD FILE ----
u = mda.Universe("test.pdb")

# ---- SELECTIONS ----
dye1 = u.select_atoms("segid A")
dye2 = u.select_atoms("segid B")

dye1_atoms = u.select_atoms("segid A and name A1 A2")
dye2_atoms = u.select_atoms("segid B and name B1 B2")

print("time distance angle")

# ---- LOOP ----
for ts in u.trajectory:

    time = ts.time

    # distance (COG)
    cog1 = dye1.center_of_geometry()
    cog2 = dye2.center_of_geometry()
    dist = np.linalg.norm(cog1 - cog2)

    # angle
    v1 = dye1_atoms.positions[1] - dye1_atoms.positions[0]
    v2 = dye2_atoms.positions[1] - dye2_atoms.positions[0]

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_theta = np.dot(v1, v2)
    angle = np.degrees(np.arccos(cos_theta))

    print(f"{time:.3f} {dist:.3f} {angle:.2f}")

print("Done.")
