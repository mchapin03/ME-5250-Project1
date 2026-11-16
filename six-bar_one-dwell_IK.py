"""
Inverse kinematics: one-dwell six-bar mechanism.
Given the output (point P, or the output-link angle th6), find every crank angle th2
that produces it, in the assembly mode used for FK (s1 = +1, s2 = -1).

Method: the mechanism is 1-DOF, so IK is a 1-D root-finding problem
    f(th2) = th6(th2) - th6_target = 0,
where th6(th2) comes from the closed-form FK. Scan th2 over a full rotation to bracket
every root (sign change), then refine each bracket with Brent's method.
Same convention as the FK script (th6 : N->F, {s} at D).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

#Geometry + FK
A  = np.array([0.80, 4.74]) #joint A location
F = np.array([-5.92, -3.04]) #joint F location
D = np.array([0.0, 0.0]) #joint D location
AB, BC, CD, CM, MN, NF, FP = 2.00, 3.70, 3.70, 3.70, 5.91, 2.57, 9.43 #link lengths
alpha = np.radians(80.0) #angle BCM
S1, S2 = +1, -1 #assembly mode used throughout

def u(t): #positions in FK
    return np.array([np.cos(t), np.sin(t)])

def circle_meet(P0, r0, P1, r1, s): 
    d = np.linalg.norm(P1 - P0) #P0->P1 distance
    a = (r0**2 - r1**2 + d**2) / (2 * d) #distance from P0 to H (foot point) along the P0->P1 line 
    h2 = r0**2 - a**2 #squared perpendicular dist from P0P1 line to the point
    if h2 < 0: #circles don't meet, links don't join
        return None
    e = (P1 - P0) / d #unit vector pointing from P0 to P1
    return P0 + a * e + s * np.sqrt(h2) * np.array([-e[1], e[0]]) #start at P0, move along line towards P1 to reach H, step s·h sideways to reach X.

def fk(th2, s1=S1, s2=S2):
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

#Inverse Kinematics
def wrap(a): #wrap angle to [-pi, pi)
    return (a + np.pi) % (2*np.pi) - np.pi

def output_to_th6(P, tol=1e-3):
    """Output point P -> output angle th6. P must lie on the circle of radius FP
    about F, because P is fixed to the rigid output link."""
    r = np.linalg.norm(P - F) #radius of output path
    if abs(r - FP) > tol: #if not within a tolerance, throw an error because output is not possible
        raise ValueError(f"P is {r:.4f} from F, but must be FP = {FP} (not on the output link's circle)")
    return np.arctan2(*(F - P)[::-1]) #F->P is th6 + pi, so N->F = direction P->F

_scan = {} #cache: the output curve doesn't depend on the target
def scan_output(n_scan):
    if n_scan not in _scan:
        grid = np.linspace(0, 2*np.pi, n_scan, endpoint=False) #find theta6 at all rotations from [0, 2pi)
        _scan[n_scan] = grid, np.array([fk(t)[0][3] for t in grid])
    return _scan[n_scan] #this doesn't change regardless of what output you want, so it is stored and cached

def ik(th6_target, n_scan=3600, tang_tol=1e-6): #runs IK with a theta6 target, number of scans around the 360 rotation, and tolerance for touching roots
    """All crank angles th2 in [0, 2pi) with th6(th2) = th6_target."""
    f = lambda t: wrap(fk(t)[0][3] - th6_target) #diff between output angle at crank angle t and target. 0 at exact IK solution
    grid, th6_grid = scan_output(n_scan) #th6 over a full crank rotation
    vals = wrap(th6_grid - th6_target) #f at every scan point
    sols = [] #initialize solutions list
    for k in range(n_scan): #for each interval between scan points
        t0, t1 = grid[k], grid[k] + 2*np.pi / n_scan #bracket [t0, t1]
        f0, f1 = vals[k], vals[(k + 1) % n_scan] #wraps 360 -> 0
        if f0 == 0: #if ik lands on exact solution
            sols.append(t0)
        elif f0 * f1 < 0: #sign change: there was a solution somewhere in the interval. Find it using Brent's Method
            sols.append(brentq(f, t0, t1, xtol=1e-14))
        elif (abs(f0) < tang_tol and vals[k-1] * f0 > 0 #touching root: f dips to ~0 between an interval without changing sign
              and abs(f0) <= abs(vals[k-1]) and abs(f0) <= abs(f1)):
            sols.append(t0) #cover otherwise uncaught solutions
    sols = np.sort(np.mod(np.array(sols), 2*np.pi)) #sort solutions and map to [0, 2pi)
    keep = [t for i, t in enumerate(sols) if i == 0 or t - sols[i-1] > np.pi / n_scan] #drop duplicate solutions 
    if len(keep) > 1 and keep[0] + 2*np.pi - keep[-1] < np.pi / n_scan:
        keep.pop() #catch first/last duplication too (0 and 360 degrees)
    return np.array(keep)

def ik_from_P(P): #IK from an output point: convert P to th6, then solve
    return ik(output_to_th6(np.asarray(P, float)))

#Main
if __name__ == "__main__":
    #The solver works in th6 internally; only inputs, prints and plots use th_FN.
    to_FN = lambda t6_deg: t6_deg + 180 #th6 (deg) -> th_FN (deg); exact for this mechanism's range
    to_th6 = lambda fn_deg: fn_deg - 180 #th_FN (deg) -> th6 (deg)
 
    #output range over the full crank rotation (from FK)
    th2s = np.radians(np.arange(0, 360, 0.1)) #build input vector
    th6s = np.array([fk(t)[0][3] for t in th2s]) #find output vector - targets outside this have no solution
    lo, hi = th6s.min(), th6s.max() #find max/min (th6, rad)
    lo_deg, hi_deg = to_FN(np.degrees(lo)), to_FN(np.degrees(hi)) #reachable range as th_FN in degrees
    print(f"Reachable output range: th_FN in [{lo_deg:.2f}, {hi_deg:.2f}] deg\n") #show reachable output range
 
    #ask the user for a target output angle within the reachable range
    while True: #keep asking until the input is valid
        try:
            fn_deg = float(input(f"Enter target output angle th_FN ({lo_deg:.2f} to {hi_deg:.2f} deg): "))
            if lo_deg - 0.005 <= fn_deg <= hi_deg + 0.005: #allow for the rounding in the printed range
                break
            print(f"Error: {fn_deg:g} deg is outside the reachable range [{lo_deg:.2f}, {hi_deg:.2f}] deg. Try again.")
        except ValueError: #input wasn't a number
            print("Error: please enter a number.")
    th6_user = np.clip(np.radians(to_th6(fn_deg)), lo, hi) #convert to th6 for the solver, kept exactly within range
    targets = {f"th_FN = {fn_deg:g} deg (user input)": th6_user}
 
    for name, t6 in targets.items():
        sols = ik(t6) #solve ik
        print(f"{name}:  ->  {len(sols)} solution(s)") #print label and num solutions
        for s in sols:
            qp, P = fk(s) #check FK of each solution
            err = np.degrees(abs(wrap(qp[3] - t6))) #FK output of this IK solution vs. the target
            print(f"    th_AB = {np.degrees(s):7.2f} deg   P = {P.round(3)}   |th_FN error| = {err:.1e} deg") #print differences
        print()
 
    #where 4 solutions exist: target values between the two small extremes inside the dwell
    grid6 = np.linspace(lo, hi, 2000)[1:-1] #target values across the range (endpoints dropped: 1 solution there)
    four = grid6[[len(ik(t, n_scan=1800)) == 4 for t in grid6]] #keep the targets that have 4 solutions
    print(f"4 solutions for th_FN in ~ [{to_FN(np.degrees(four.min())):.2f}, {to_FN(np.degrees(four.max())):.2f}] deg (inside the dwell); "
          "2 everywhere else in the range\n")
 
    #Plotting (all output angles shown as th_FN)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.degrees(th2s), to_FN(np.degrees(th6s)), "k-", lw=1.5, label="θFN(θAB) from FK")
    t6 = th6_user #user target
    for s in ik(t6):
        ax[0].plot(np.degrees(s), to_FN(np.degrees(t6)), "ro")
    ax[0].axhline(to_FN(np.degrees(t6)), color="r", ls="--", lw=1, label=f"target θFN = {fn_deg:g}°")
    ax[0].set_xlabel("θAB (deg)"); ax[0].set_ylabel("θFN (deg)")
    ax[0].set_title("IK = where the output curve meets the target"); ax[0].legend(); ax[0].grid()
    #zoom on the dwell (th_AB from -60 to 60 deg) with the 4-solution target
    zz = np.radians(np.arange(-60, 60.01, 0.1)) #dwell zoom range
    ax[1].plot(np.degrees(zz), [to_FN(np.degrees(fk(t)[0][3])) for t in zz], "k-", lw=1.5)
    t6 = np.radians(to_th6(89.3)) #fixed example inside the 4-solution band (th_FN = 89.3 deg)
    for s in ik(t6):
        if abs(wrap(s)) < np.radians(60): #the 4th solution (~184 deg) is off-panel
            ax[1].plot(np.degrees(wrap(s)), 89.3, "ro")
    ax[1].axhline(89.3, color="r", ls="--", lw=1, label="target θFN = 89.3°")
    ax[1].set_xlabel("θAB (deg, wrapped)")
    ax[1].set_ylabel("θFN (deg)")
    ax[1].set_title("Dwell zoom: 3 of the 4 solutions (4th at θAB ≈ 184°)")
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("ik_sixbar.png", dpi=120)