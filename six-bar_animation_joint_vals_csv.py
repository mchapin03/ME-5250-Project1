"""
Animation of the one-dwell six-bar over one full crank rotation.
Imports the FK solver, draws the mechanism + joint-range arcs at every crank angle,
saves each frame as a PNG, then stacks the frames into a 30 fps MP4.

Also writes a CSV of every joint angle (th2..th6, CCW from +x) for th2 = 0-360 deg, and
reports the dwell: the crank range over which th6 stays within +-0.5 deg of its dwell value.
"""
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg") #draw off-screen: no windows pop up for 360 frames
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil #for deleting the frames folder

#Import FK script
HERE = os.path.dirname(os.path.abspath(__file__)) #this folder
FK_PATH = os.path.join(HERE, "six-bar_one-dwell_closed-form_FK.py") #path
spec = importlib.util.spec_from_file_location("sixbar_fk", FK_PATH) #spec module
fkm = importlib.util.module_from_spec(spec) #import as module
spec.loader.exec_module(fkm) #load module

A, D, F, alpha = fkm.A, fkm.D, fkm.F, fkm.alpha

#Animation settings
FPS = 30 #will give exactly 12s video from 360 frames
FRAME_DIR = os.path.join(HERE, "frames") #folder to store frames
VIDEO_PATH = os.path.join(HERE, "sixbar_rotation.mp4") #video path
S1, S2 = +1, -1 #assembly mode: same as FK script
R_ARC = 0.8 #radius of the joint-range arcs: same as FK script
KEEP_FRAMES = False #set True to keep the individual frame PNGs after the video is made

# ---------------- solve FK once for the whole rotation ----------------
# Sweep 0..360 deg (361 poses, as in the FK script) so the joint ranges match it exactly.
# The video uses only 0..359 (360 frames = 12 s at 30 fps): 360 deg is the same pose
# as 0 deg, so leaving it out makes the video loop without a repeated frame.
th2s = np.radians(np.arange(0, 361, 1)) #sweep through full crank rotation
N_FRAMES = 360 #number of video frames (0-359 deg; the sweep has 361 poses)
QP, PS = [], [] #solutions lists
for t in th2s: #loop through all crank rotations
    r = fkm.fk(t, S1, S2) #run FK
    assert r is not None, "loop failed to close: branch/toggle problem" #error if fk fails
    assert np.max(np.abs(fkm.g(t, r[0]))) < 1e-9 #loop closures satisfied
    QP.append(r[0]); PS.append(r[1]) #append solutions
QP = np.unwrap(np.array(QP), axis=0) #unwrap
PS = np.array(PS) #convert to numpy array
T2 = th2s #theta2
T3, T4, T5, T6 = QP.T #passive theta values

#joint positions at every frame
JOINTS = [fkm.joints_at(T2[i], T3[i], T5[i]) for i in range(N_FRAMES)]   # (jB, jC, jM, jN)

#joint ranges (same for every frame)
zero = np.zeros_like(T2)
range_defs = { #(reference link direction, moving link direction)
    "A": (zero,        T2), #ground -> crank AB
    "B": (T2 + np.pi,  T3), #BA -> BC
    "C": (T3 + np.pi,  T4), #CB (rigid link) -> CD
    "D": (zero,        T4 + np.pi), #ground -> DC
    "M": (T3 + alpha,  T5), #MC (rigid link) -> MN
    "N": (T5 + np.pi,  T6), #NM -> NF
    "F": (zero,        T6 + np.pi), #ground -> FN (output link)
}
RANGES = {} #dictionary of joint ranges
for name, (ref, mov) in range_defs.items():
    rel = np.unwrap(mov - ref) #joint angle over the rotation
    RANGES[name] = (rel.min(), rel.max(), ref)
COLORS = dict(zip(range_defs, plt.cm.tab10(np.arange(len(range_defs))))) #colormap for joint spans

#fix axis limits so it is the same for every frame
allpts = np.vstack([A, D, F, PS] + [np.array(j) for j in JOINTS])
pad = R_ARC + 0.5
XLIM = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
YLIM = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

#draw a frame
def draw_frame(ax, i):
    ax.clear()
    jB, jC, jM, jN = JOINTS[i] #joint positions
    jP = PS[i] #output point position
    pos = {"A": A, "B": jB, "C": jC, "D": D, "M": jM, "N": jN, "F": F} #lookup by name

    #links
    ax.plot(*np.c_[A, jB], "k-", lw=3) #crank AB
    ax.fill(*np.c_[jB, jC, jM], color="0.85", ec="k", lw=3) #rigid link BCM
    ax.plot(*np.c_[D, jC], "k-", lw=3) #rocker DC
    ax.plot(*np.c_[jM, jN], "k-", lw=3) #link MN
    ax.plot(*np.c_[F, jP], "k-", lw=3) #output FNP
    ax.plot(PS[:, 0], PS[:, 1], "-", color="tab:blue", lw=1, alpha=0.6, label="path of P") #path of P plot

    #joint range arcs (reference link frozen at this frame's pose)
    for name, (lo, hi, ref) in RANGES.items(): #for each joint
        span = np.degrees(hi - lo) #span of joint
        if span < 360: #make an arc if it is not a full rotation
            arc = np.linspace(lo, hi, 200) + ref[i]
            txt = f"{span:.0f} degrees"
        else: #otherwise make a circle
            arc = np.linspace(0, 2*np.pi, 200)
            txt = "360 deg (full rotation)"
        p = pos[name] #this joint's position
        ax.plot(p[0] + R_ARC*np.cos(arc), p[1] + R_ARC*np.sin(arc), color=COLORS[name], lw=2.5, label=f"joint {name}: {txt}") #plot the arc
        ax.plot(*p, "s" if name in ("A", "D", "F") else "o", ms=9, mfc="w", mec="k", mew=2, zorder=5) #plot the points (square=grounded, circle=moving)
        ax.annotate(name, p, xytext=(-14, 8), textcoords="offset points", fontweight="bold") #label the joints

    ax.plot(*jP, "o", ms=6, color="tab:blue", zorder=5) #output point P
    ax.annotate("P", jP, xytext=(8, 4), textcoords="offset points", color="tab:blue") #label P

    #Plot settings
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.grid()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9) #legend outside, right of the plot
    ax.set_title(f"Mechanism at θ2 = {np.degrees(T2[i]):.0f}°, arcs = joint range over full crank rotation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

#dwell: input range with output within +-0.5 deg of the dwell value
DWELL_TOL = 0.5 # dwell tolerance in deg, from project definition
def find_dwell(step_deg=0.01):
    """Fine sweep of th6 (0.01 deg steps). Dwell value = midpoint of the output's small
    wiggle inside the dwell. Dwell range = the crank range around it with |th6 - value| <= tol."""

    th2f = np.radians(np.arange(0, 360, step_deg)) #fine grid of crank angles (0.01 deg steps), in radians
    th6f = np.degrees(np.unwrap([fkm.fk(t, S1, S2)[0][3] for t in th2f])) #output angle th6 at each, unwrapped, in degrees
    n = len(th6f) #number of values
    prev, nxt = np.roll(th6f, 1), np.roll(th6f, -1) #cyclic neighbours
    ext = np.where(((th6f > prev) & (th6f >= nxt)) | ((th6f < prev) & (th6f <= nxt)))[0] #get all extremes (including output range endpoints)
    ext = [k for k in ext if th6f[k] not in (th6f.max(), th6f.min())]   # dwell extremes only (direction switches)
    if ext:
        value = (th6f[ext].max() + th6f[ext].min()) / 2 #find midpoint
        k0 = ext[0]
    else: #no wiggle: use flattest point
        k0 = int(np.argmin(np.abs(nxt - prev))) #flattest point - smallest change between neighbours
        value = th6f[k0]
    inside = np.abs(th6f - value) <= DWELL_TOL #true wherever the output is within tolerance
    lo_k, hi_k = k0, k0 #both ends of the dwell start at k0, then grow outwards
    while inside[(lo_k - 1) % n] and (k0 - lo_k) < n: lo_k -= 1   #grow backwards (wraps 0 -> 360)
    while inside[(hi_k + 1) % n] and (hi_k - lo_k) < n: hi_k += 1 #grow forwards
    start = (lo_k * step_deg) % 360
    end = (hi_k * step_deg) % 360
    span = (hi_k - lo_k) * step_deg #find span of dwell range
    resid = th6f[np.arange(lo_k, hi_k + 1) % n]
    return value, start, end, span, resid.max() - resid.min()

def in_dwell(th2_deg, start, end):
    """True if th2 lies in the dwell range [start, end] (which may wrap past 360)."""
    th2_deg = th2_deg % 360
    return (start <= th2_deg <= end) if start <= end else (th2_deg >= start or th2_deg <= end)

#Render frames, then build the video
if __name__ == "__main__":
    #dwell range
    dw_value, dw_start, dw_end, dw_span, dw_resid = find_dwell() #get dwell values
    print(f"Dwell value: th6 = {dw_value:.3f} deg") #report dwell values
    print(f"Output within +-{DWELL_TOL} deg of it for th2 from {dw_start:.2f} to {dw_end:.2f} deg "
          f"(through 0 deg if start > end): dwell angle = {dw_span:.2f} deg of crank rotation")
    print(f"Max residual output motion inside the dwell: {dw_resid:.3f} deg\n") #residual is 1 by construction (-0.5 to 0.5 degrees), prints 0.9999 due to 0.01 grid resolution

    #CSV of all joints
    wrap_deg = lambda a: (np.degrees(a) + 180) % 360 - 180 #report angles in [-180, 180)
    csv_path = os.path.join(HERE, "sixbar_joint_angles.csv") #where to save
    th2_deg_all = np.degrees(T2) #full 360 sweep
    rows = np.column_stack([th2_deg_all, wrap_deg(T3), wrap_deg(T4), wrap_deg(T5), wrap_deg(T6),
                            th2_deg_all, wrap_deg(T4 + np.pi), wrap_deg(T6 + np.pi), #ground joints A, D, F
                            PS[:, 0], PS[:, 1], [in_dwell(t, dw_start, dw_end) for t in th2_deg_all]]) #setup csv
    header = ("th2_deg (A->B),th3_deg (B->C),th4_deg (C->D),th5_deg (M->N),th6_deg (N->F),"
              "joint_A_deg (ground->AB),joint_D_deg (ground->DC),joint_F_deg (ground->FN),"
              "P_x,P_y,in_dwell (1 = output within +-0.5 deg of dwell value)")
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="",
               fmt=["%.0f", "%.4f", "%.4f", "%.4f", "%.4f",
                    "%.0f", "%.4f", "%.4f",
                    "%.4f", "%.4f", "%d"]) #save csv
    print(f"Saved joint angles for {len(T2)} crank angles to {csv_path}\n")

    os.makedirs(FRAME_DIR, exist_ok=True) #create the frames folder
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120) #set figure
    fig.subplots_adjust(left=0.06, right=0.78, top=0.94, bottom=0.07) #shrink the axes so the outside legend fits in the frame
    frame_files = [] #frames to stack
    for i in range(N_FRAMES): #go through all frames
        draw_frame(ax, i) #draw the frame
        path = os.path.join(FRAME_DIR, f"frame_{i:03d}.png") #name to save the frame
        fig.savefig(path, dpi=120) #save the frame
        frame_files.append(path) #add to frames list
        if i % 30 == 0: #report rendered frames every 30 in case it stalls
            print(f"rendered frame {i}/{N_FRAMES}")
    plt.close(fig) #free the memory

    with imageio.get_writer(VIDEO_PATH, fps=FPS) as writer: #write a video
        for path in frame_files:
            writer.append_data(imageio.imread(path)) #stack frames
    print(f"\nSaved video ({len(frame_files) / FPS:.0f} s at {FPS} fps) to {VIDEO_PATH}") #successfully made video
    if not KEEP_FRAMES:
        shutil.rmtree(FRAME_DIR) #delete the frames folder and every PNG in it
        print(f"Deleted {FRAME_DIR} (set KEEP_FRAMES = True to keep the frames)")
    else:
        print(f"Kept {len(frame_files)} frames in {FRAME_DIR}")