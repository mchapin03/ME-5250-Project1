"""
D6 Velocity kinematics, statics and singularities: one-dwell six-bar mechanism.
Imports the FK solver from "six-bar_one-dwell_closed-form_FK.py" (same folder), so 
D6 uses exactly the D3 geometry, frame and assembly mode (s1 = +1, s2 = -1).

Report convention: theta_IJ = direction from joint I to joint J, CCW from +x, {s} at D, mm.
q = (th_AB; th_BC, th_DC, th_MN, th_FN), beta = pi + alpha = 260 deg
The FK script uses th4 = C->D and th6 = N->F, so th_DC = th4 + pi and th_FN = th6 + pi.

Outputs: printed FD check and singularity classification, D5_values.csv,
D6_mechanical_advantage.png
"""
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg") #save figures without opening windows (works under run_all.py)
import matplotlib.pyplot as plt

#Import FK script
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sixbar_fk", os.path.join(HERE, "six-bar_one-dwell_closed-form_FK.py"))
fkm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fkm)

S1, S2 = +1, -1 #assembly mode used throughout
L_AB, L_BC, L_CM, L_DC, L_MN, L_FN = fkm.AB, fkm.BC, fkm.CM, fkm.CD, fkm.MN, fkm.NF #link lengths (mm)
beta = np.pi + fkm.alpha #C->M direction = th_BC + beta (260 deg)
DWELL = (327.48, 29.70) #dwell range of th_AB from D7 (deg), wraps through 0

def q_report(th_AB):
    """FK in the report convention: returns (th_AB, th_BC, th_DC, th_MN, th_FN) in rad."""
    (th3, th4, th5, th6), _ = fkm.fk(th_AB, S1, S2)
    return np.array([th_AB, th3, th4 + np.pi, th5, th6 + np.pi])

def H_matrices(q):
    """Ha = dg/dq_a (4x1) and Hp = dg/dq_p (4x4) from Eqs. (2)-(5)."""
    th_AB, th_BC, th_DC, th_MN, th_FN = q
    Ha = L_AB * np.array([-np.sin(th_AB), np.cos(th_AB), -np.sin(th_AB), np.cos(th_AB)])
    Hp = np.array([
        [-L_BC*np.sin(th_BC),                             L_DC*np.sin(th_DC),  0,                   0],
        [ L_BC*np.cos(th_BC),                            -L_DC*np.cos(th_DC),  0,                   0],
        [-L_BC*np.sin(th_BC) - L_CM*np.sin(th_BC + beta), 0,                  -L_MN*np.sin(th_MN),  L_FN*np.sin(th_FN)],
        [ L_BC*np.cos(th_BC) + L_CM*np.cos(th_BC + beta), 0,                   L_MN*np.cos(th_MN), -L_FN*np.cos(th_FN)]])
    return Ha, Hp

if __name__ == "__main__":
    #Sweep the crank: build Ha and Hp, compute J
    thdeg = np.round(np.arange(0, 360.0001, 0.1), 1) #crank angles (deg), 3601 points
    th = np.radians(thdeg)
    n = len(th)
    J, detHp, sv, thFN, gamC, gamN = (np.zeros(n) for _ in range(6))
    for i, t in enumerate(th):
        q = q_report(t)
        Ha, Hp = H_matrices(q)
        x = np.linalg.solve(Hp, Ha) #Hp^-1 Ha
        J[i] = -x[3] #J = -[0 0 0 1] Hp^-1 Ha
        detHp[i] = np.linalg.det(Hp)
        sv[i] = np.linalg.svd(np.column_stack([Ha, Hp]), compute_uv=False).min() #~0 -> configuration-space singularity
        thFN[i] = q[4]
        gamC[i] = np.degrees(np.arccos(np.cos(q[1] - q[2]))) #transmission angle at C (coupler BC vs rocker DC)
        gamN[i] = np.degrees(np.arccos(np.cos(q[3] - q[4]))) #transmission angle at N (dyad MN vs output FN)
    thFNdeg = np.degrees(np.unwrap(thFN))

    #Check J against finite differences of the FK
    print("Finite-difference check\n  theta_AB      J         J_fd       error")
    h, fd_err = 1e-6, 0.0
    for t in range(15, 346, 30):
        qp, qm = q_report(np.radians(t) + h), q_report(np.radians(t) - h)
        Jfd = np.angle(np.exp(1j * (qp[4] - qm[4]))) / (2 * h) #central difference of th_FN
        Jan = J[int(round(t / 0.1))]
        print(f"  {t:6.1f}   {Jan:9.5f}  {Jfd:9.5f}   {abs(Jan - Jfd):.1e}")
        fd_err = max(fd_err, abs(Jan - Jfd))
    print(f"  Max |J - J_fd| over 12 angles = {fd_err:.1e}")

    #Find and classify singularities
    print("\nSingularities")
    print(f"  Configuration-space: smallest singular value of [Ha Hp] = {sv.min():.1f} mm (none unless ~0)")
    iA = np.where(detHp[:-1] * detHp[1:] < 0)[0] #det Hp changes sign
    if len(iA) == 0:
        print("  Actuator: none (det Hp never reaches zero)")
    for k in iA:
        print(f"  Actuator: theta_AB = {thdeg[k]:.1f} deg")
    print(f"  Transmission angle at C: {gamC.min():.0f} to {gamC.max():.0f} deg;  at N: {gamN.min():.0f} to {gamN.max():.0f} deg")

    iE = np.where(J[:-1] * J[1:] < 0)[0] #J changes sign
    ee_ang, ee_val = [], []
    for k in iE:
        if np.any(np.abs(k - iA) <= 1): #J jumps through infinity there, not zero
            continue
        t0 = thdeg[k] - J[k] * (thdeg[k+1] - thdeg[k]) / (J[k+1] - J[k]) #interpolate J = 0
        ee_ang.append(t0); ee_val.append(np.degrees(q_report(np.radians(t0))[4])) #th_FN at the interpolated angle
        print(f"  End-effector (J = 0): theta_AB = {t0:.1f} deg, theta_FN = {ee_val[-1]:.2f} deg")

    #Mechanical advantage: tau_AB = J T_out -> MA = 1/|J|
    MA = 1 / np.abs(J)
    i_min = np.argmin(MA)
    print(f"\nLowest MA = {MA[i_min]:.2f} at theta_AB = {thdeg[i_min]:.1f} deg (MA -> infinity where J = 0)")
    in_dwell = (thdeg >= DWELL[0]) | (thdeg <= DWELL[1])
    print(f"Lowest MA inside the dwell = {MA[in_dwell].min():.2f}")

    #Table of the values in the D5 plot
    np.savetxt(os.path.join(HERE, "D5_values.csv"), np.column_stack([thdeg, np.round(thFNdeg, 2)]),
               delimiter=",", header="theta_AB_deg,theta_FN_deg", comments="", fmt=["%.1f", "%.2f"])
    print("\nOutput angle every 15 deg of crank rotation:\n  theta_AB_deg  theta_FN_deg")
    for k in range(0, n, 150):
        print(f"  {thdeg[k]:12.1f}  {thFNdeg[k]:12.2f}")
    print("\nSingularities to be marked on the D5 plot:\n  theta_AB_deg  theta_FN_deg")
    for a, v in zip(ee_ang, ee_val):
        print(f"  {a:12.1f}  {v:12.2f}")

    np.savetxt(os.path.join(HERE, "D6_singularities.csv"), np.column_stack([ee_ang, ee_val]),
               delimiter=",", header="theta_AB_deg,theta_FN_deg", comments="", fmt="%.2f") #for the D5 plot
    
    #Plot the mechanical advantage vs input
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(thdeg, MA, "k", lw=1.5)
    ax.set_ylim(0.5 * MA.min(), 1e3); ax.set_xlim(0, 360)
    for a in ee_ang:
        ax.axvline(a, color="r", ls="--", lw=1)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_xlabel("Crank angle θAB (deg)")
    ax.set_ylabel("|MA| = 1/|J|")
    ax.set_title("Mechanical advantage vs input (red: J = 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "D6_mechanical_advantage.png"), dpi=200)