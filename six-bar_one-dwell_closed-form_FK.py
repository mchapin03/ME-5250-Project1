"""
D3 Forward kinematics: one-dwell six-bar mechanism.
Convention (Head-tail):
  th1  : D->A (const)    th1' : D->F (const)
  th2  : A->B (q_a)      th3  : B->C        th4 : C->D
  th5  : M->N            th6  : N->F        alpha = BCM angle (constant)
  C->M direction = th3 + pi + alpha  (BCM is one rigid link, pi reverses direction)
{s} frame: origin at D, x right, y up, angles CCW from +x.
"""
import numpy as np
import matplotlib.pyplot as plt

# Mechanism Geometry
# Mechanism Geometry (dimensions defined in inches, converted to mm)
IN_TO_MM = 25.4 #exact
A  = np.array([0.80, 4.74]) * IN_TO_MM #A coordinates in D ({s}), mm
F  = np.array([-5.92, -3.04]) * IN_TO_MM #F coordinates in D ({s}), mm
D  = np.array([0.0, 0.0]) #we define as origin
AB, BC, CD, CM, MN, NF, FP = np.array([2.00, 3.70, 3.70, 3.70, 5.91, 2.57, 9.43]) * IN_TO_MM #link lengths, mm
alpha = np.radians(80.0) #angle BCM inside rigid link
DA, th1  = np.hypot(*A), np.arctan2(A[1], A[0]) #Define DA (ground) length and DA angle (theta1, constant)
DF, th1p = np.hypot(*F), np.arctan2(F[1], F[0]) #Define DF (ground) length and DF angle (theta1', constant)

def u(t): #unit vector at angle t
    return np.array([np.cos(t), np.sin(t)])

def circle_meet(P0, r0, P1, r1, s): # (joint location 1, link length 1, joint loc 2, link len 2, elbow up/down)
    """Point at distance r0 from P0 and r1 from P1. s = +1/-1 picks the branch
    (side of line P0->P1). Returns None if the loop cannot close."""
    d = np.linalg.norm(P1 - P0) #P0->P1 distance
    a = (r0**2 - r1**2 + d**2) / (2 * d) #distance from P0 to H (foot point) along the P0->P1 line 
    h2 = r0**2 - a**2 #squared perpendicular dist from P0P1 line to the point
    if h2 < 0: #circles don't meet, links don't join
        return None
    e = (P1 - P0) / d #unit vector pointing from P0 to P1
    return P0 + a * e + s * np.sqrt(h2) * np.array([-e[1], e[0]]) #start at P0, move along line towards P1 to reach H, step s·h sideways to reach X.

def fk(th2, s1, s2):
    """Given q_a = th2 and branch signs, return q_p = (th3..th6) and P."""
    B = A + AB * u(th2) #location of joint B
    C = circle_meet(B, BC, D, CD, s1) #solve for joint C location
    if C is None: return None #loop 1 can't close at this input
    th3 = np.arctan2(*(C - B)[::-1]) #find theta 3
    th4 = np.arctan2(*(D - C)[::-1]) #find theta 4
    M = C + CM * u(th3 + np.pi + alpha) #location of joint M
    N = circle_meet(M, MN, F, NF, s2) #solve for joint N location
    if N is None: return None #loop 2 can't close at this input
    th5 = np.arctan2(*(N - M)[::-1]) #find theta 5
    th6 = np.arctan2(*(F - N)[::-1]) #find theta 6
    P = F - FP * u(th6) #find P (output point) location
    return np.array([th3, th4, th5, th6]), P #return solutions

def g(th2, qp): #compute loop closure equations, should all result in 0
    th3, th4, th5, th6 = qp
    l1 = DA*u(th1) + AB*u(th2) + BC*u(th3) + CD*u(th4) #creates loop 1 equations (u() = cos, sin for both x and y)
    l2 = -CD*u(th4) - CM*u(th3+alpha) + MN*u(th5) + NF*u(th6) - DF*u(th1p) #creates loop 2 equations (u() = cos, sin for both x and y)
    return np.concatenate([l1, l2])

def joints_at(th2, th3, th5): #joint positions for one pose (used for joint range diagram)
    jB = A + AB * u(th2)
    jC = jB + BC * u(th3)
    jM = jC + CM * u(th3 + np.pi + alpha)
    jN = jM + MN * u(th5)
    return jB, jC, jM, jN


if __name__ == "__main__":
    while True: #keep asking until the input is valid
        try:
            th2_deg = float(input("Enter crank angle th2 (0-360 deg): "))
            if 0 <= th2_deg <= 360:
                break
            print("Please enter a value between 0 and 360.")
        except ValueError: #input wasn't a number
            print("Please enter a number.")
    th2_test = np.radians(th2_deg) #input angle
    print(f"Assembly modes at th2 = {th2_deg:g} deg:")
    for s1 in (+1, -1): #for elbow-up/down on C
        for s2 in (+1, -1): #for elbow-up/down on N
            r = fk(th2_test, s1, s2) #solve FK
            if r: print(f"  s1={s1:+d} s2={s2:+d}  q_p(deg)={np.degrees(r[0]).round(1)}  P={r[1].round(2)}") #prints only the assembly modes that exist at this input

    #sweep the full crank rotation
    S1, S2 = +1, -1 #assembly mode we used
    th2s = np.radians(np.arange(0, 361, 1)) #build sweep
    QP, PS = [], [] #results lists
    for t in th2s: #repeat for every angle
        r = fk(t, S1, S2) #solve FK
        assert r is not None, "loop failed to close: branch/toggle problem" #loop must close at every input, otherwise the branch hit a toggle
        QP.append(r[0]); PS.append(r[1]) #log results
        assert np.max(np.abs(g(t, r[0]))) < 1e-9 #check the loop closures are 0
    QP = np.unwrap(np.array(QP), axis=0) #unwraps atan2
    PS = np.array(PS) #convert to np array
    print(f"\nBranch s1={S1}, s2={S2}: all 361 poses close, max |g| < 1e-9") #prints confirmed answer
    print(f"th_FN range: {np.degrees(QP[:,3].min()) + 180:.1f} to {np.degrees(QP[:,3].max()) + 180:.1f} deg") #reports output angle range

    #Plotting
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(PS[:, 0], PS[:, 1])
    ax[0].set_aspect("equal")
    ax[0].set_title("Output path of P Relative to Origin at D")
    ax[0].set_xlabel("x (mm)")
    ax[0].set_ylabel("y (mm)")
    #report convention: th_DC = th4 (C->D) + 180, th_FN = th6 (N->F) + 180
    report_angles = {"θBC": np.degrees(QP[:, 0]),
                     "θDC": np.degrees(QP[:, 1]) + 180,
                     "θMN": np.degrees(QP[:, 2]),
                     "θFN": np.degrees(QP[:, 3]) + 180}
    for name, vals in report_angles.items():
        ax[1].plot(np.degrees(th2s), vals, label=name)
    ax[1].set_xlabel("θAB (deg)")
    ax[1].set_ylabel("angle (deg)")
    ax[1].set_title("Passive joints vs input")
    ax[1].legend()
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()
    plt.savefig("fk_sixbar.png", dpi=120)


    #Diagram of joint ranges in +1, -1 assembly mode
    '''
    Draws the linkage at the user's input angle, then puts an arc at every joint showing
    that joint's range over the full crank sweep. Each arc shows the second link's swing
    relative to the first link, with the first link frozen at the drawn pose.
    '''

    i0 = int(round(th2_deg)) #reference pose (use user input for position shown, does not change joint ranges over sweep)
    T2 = th2s #theta2 at every sweep step
    T3, T4, T5, T6 = QP.T #unwrapped passive angles at every step

    jB, jC, jM, jN = joints_at(T2[i0], T3[i0], T5[i0]) #get joint locations
    jP = PS[i0] 

    zero = np.zeros_like(T2) #reference direction for ground pivots
    joint_defs = {
        "A": (A,  zero,            T2),           #ground -> crank AB
        "B": (jB, T2 + np.pi,      T3),           #BA -> BC
        "C": (jC, T3 + np.pi,      T4),           #CB (rigid link) -> CD
        "D": (D,  zero,            T4 + np.pi),   #ground -> DC
        "M": (jM, T3 + alpha,      T5),           #MC (rigid link) -> MN
        "N": (jN, T5 + np.pi,      T6),           #NM -> NF
        "F": (F,  zero,            T6 + np.pi),   #ground -> FN (output link)
    }

    #Plotting setup
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    # draw links
    ax2.plot(*np.c_[A, jB], "k-", lw=3) #crank AB
    ax2.fill(*np.c_[jB, jC, jM], color="0.85", ec="k", lw=3) #rigid link BCM
    ax2.plot(*np.c_[D, jC], "k-", lw=3) #rocker DC
    ax2.plot(*np.c_[jM, jN], "k-", lw=3) #link MN
    ax2.plot(*np.c_[F, jP], "k-", lw=3) #output FNP
    ax2.plot(PS[:, 0], PS[:, 1], "-", color="tab:blue", lw=1, alpha=0.6, label="path of P") #draw path of P (already calculated)

    #joint range arcs
    r_arc = 0.8 * IN_TO_MM #radius of arc, arbitrary (mm)
    colors = plt.cm.tab10(np.arange(len(joint_defs))) #use a colormap
    for (name, (pos, ref, mov)), col in zip(joint_defs.items(), colors):
        rel = np.unwrap(mov - ref) #joint angle over the sweep
        lo, hi = rel.min(), rel.max() #min and max
        span = np.degrees(hi - lo) #span of joint angles
        base = ref[i0] #freeze reference link at the drawn pose for showing
        arc = np.linspace(lo, hi, 200) + base if span < 360 else np.linspace(0, 2*np.pi, 200) #create arc vector
        txt = "360 deg (full rotation)" if span >= 360 else f"{span:.0f} degrees" #show text of spans 
        ax2.plot(pos[0] + r_arc*np.cos(arc), pos[1] + r_arc*np.sin(arc), color=col, lw=2.5, label=f"joint {name}: {txt}") #plot the joint's range arc
        ground = name in ("A", "D", "F") #define grounded joints
        ax2.plot(*pos, "s" if ground else "o", ms=9, mfc="w", mec="k", mew=2, zorder=5) #squares are grounded R joints, circles are moving R joints
        ax2.annotate(name, pos, xytext=(-14, 8), textcoords="offset points", fontweight="bold") #label the joint
    ax2.plot(*jP, "o", ms=6, color="tab:blue", zorder=5) #show output point P
    ax2.annotate("P", jP, xytext=(8, 4), textcoords="offset points", color="tab:blue") #label output point P

    #Plotting
    ax2.set_aspect("equal")
    ax2.grid()
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9) #legend outside, right of the plot
    ax2.set_title(f"Mechanism at θAB = {np.degrees(T2[i0]):.0f}°, arcs = joint range over full crank rotation")
    ax2.set_xlabel("x (mm)") 
    ax2.set_ylabel("y (mm)")
    plt.tight_layout()
    fig2.savefig("fk_sixbar_ranges.png", dpi=120)