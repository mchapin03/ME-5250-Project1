"""
Conversion of Matt Chapin's "Data Analysis.ipynb" for the use of running 
through "run_all.py". Claude was used to convert from the original notebook file. 
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") #save figures without opening windows (works under run_all.py)
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = [10, 5]

HERE = os.path.dirname(os.path.abspath(__file__)) #LivePlotter folder
ROOT = os.path.dirname(HERE) #project folder, where the other scripts write their outputs

### Read the data
jointDF = pd.read_csv(
    os.path.join(HERE, "data.txt"),
    sep=r"[:,]\s*",
    engine="python",
    header=None,
    usecols=[1, 3, 5],
    names=["A", "D", "F"],
)
calcDF = pd.read_csv(os.path.join(ROOT, "sixbar_joint_angles.csv"), header=0)

def column(df, *names):
    """Return the first column that exists, so either CSV header version works."""
    for n in names:
        if n in df.columns:
            return df[n].to_numpy()
    raise KeyError(f"None of {names} found in sixbar_joint_angles.csv")

### Make data fit to each other + calculate error
start = 0
end = start + 360

shift = 100 #offset (deg) between the encoder's zero at joint A and th_AB = 0
calcA = column(calcDF, 'joint_A_deg (ground->AB)', 'th_AB_deg (A->B; joint A)')
calcD = column(calcDF, 'joint_D_deg (ground->DC)', 'th_DC_deg (D->C; joint D)')
calcF = column(calcDF, 'joint_F_deg (ground->FN)', 'th_FN_deg (F->N; joint F)')
calcA, calcD, calcF = calcA[:-1], calcD[:-1], calcF[:-1] #drop the 360 deg row (duplicate of 0 deg)

def model(y, a):
    """Calculated joint angle at encoder reading a (deg), zeroed at the starting pose like the encoders.
    period=360 wraps th_AB = a - shift around 0/360 instead of rolling the array."""
    return np.interp(a - shift, calcA, y, period=360) - np.interp(-shift, calcA, y, period=360)

errAD = model(calcD, jointDF['A']) - jointDF['D']
errAF = model(calcF, jointDF['A']) - jointDF['F']

inRange = jointDF['A'].between(0, 360)
MSE_AD = (errAD[inRange]**2).mean()
MSE_AF = (errAF[inRange]**2).mean()
print(f"MSE_AD = {MSE_AD:.4f} deg^2, MSE_AF = {MSE_AF:.4f} deg^2")
print(f"RMSE_AD = {np.sqrt(MSE_AD):.3f} deg, RMSE_AF = {np.sqrt(MSE_AF):.3f} deg")
print(f"Error range: joint D {errAD[inRange].min():.2f} to {errAD[inRange].max():.2f} deg, "
      f"joint F {errAF[inRange].min():.2f} to {errAF[inRange].max():.2f} deg")

#errors at evenly spaced inputs, for the D8 comparison table
print("\n  Joint A [deg]   err D [deg]   err F [deg]")
for a in range(0, 360, 60):
    k = (jointDF['A'][inRange] - a).abs().idxmin() #closest measured sample within 0-360
    print(f"  {jointDF['A'][k]:13.1f}   {errAD[k]:11.2f}   {errAF[k]:11.2f}")

### Measured dwell: longest contiguous run of joint A where measured F stays within +-0.5 deg of centralAngle
centralAngle = 28.94
offset = 0.5
mA = jointDF['A'][inRange].to_numpy()
mF = jointDF['F'][inRange].to_numpy()
inside = np.abs(mF - centralAngle) <= offset
best, run_start = (0, 0), None
for k, ok in enumerate(np.r_[inside, False]): #trailing False closes the last run
    if ok and run_start is None:
        run_start = k
    elif not ok and run_start is not None:
        if k - run_start > best[1] - best[0]:
            best = (run_start, k)
        run_start = None
a0, a1 = mA[best[0]], mA[best[1] - 1]
print(f"\nMeasured dwell: joint F within +-{offset} deg of {centralAngle} deg for joint A from "
      f"{a0:.1f} to {a1:.1f} deg ({a1 - a0:.1f} deg of crank rotation)")

### Measured vs calculated, and error
aa = np.arange(start, end + 0.5, 0.5) #encoder readings to draw the calculated curves at
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax1.plot(jointDF['A'], jointDF['D'], color='#BA3838', label="Measured")
ax1.plot(aa, model(calcD, aa), linestyle='dashdot', color='#FFBE63', label="Calculated")
ax1.minorticks_on()
ax1.grid()
ax1.grid(which='minor', alpha=0.40)
ax1.set_title("Joint A (input) vs. Joint D")
ax1.set_ylabel("Joint D Rotation [Deg]")
ax1.legend(loc='best')
ax1.set_xlim(start, end)

ax2.plot(jointDF['A'], jointDF['F'], color='#3843BA', label="Measured")
ax2.plot(aa, model(calcF, aa), linestyle='dashdot', color='#BC84FA', label="Calculated")
ax2.minorticks_on()
ax2.grid()
ax2.grid(which='minor', alpha=0.40)
ax2.set_title("Joint A (input) vs. Joint F (output)")
ax2.set_ylabel("Joint F Rotation [Deg]")
ax2.set_xlim(start, end)
ax2.legend(loc='best')

ax3.plot(jointDF['A'], errAD, color='#fccf03', label="Joint D Error")
ax3.plot(jointDF['A'], errAF, color='#2f8dcc', label="Joint F Error")
ax3.minorticks_on()
ax3.grid()
ax3.grid(which='minor', alpha=0.40)
ax3.set_title("Angle Error")
ax3.set_xlabel("Joint A Rotation [Deg]")
ax3.set_ylabel("Error [Deg]")
ax3.set_xlim(start, end)
ax3.set_ylim(-3, 3)
ax3.legend(loc='best')

plt.tight_layout(h_pad=2)
fig.savefig(os.path.join(HERE, "D8_joint_comparison.png"))
plt.close(fig)

### Measured output zoomed to the dwell
fig = plt.figure()
plt.plot(jointDF['A'], jointDF['F'], color='#3843BA')
plt.minorticks_on()
plt.grid()
plt.grid(which='minor', alpha=0.40)
plt.title("Joint A (input) vs. Joint F (output) - Zoomed to Dwell")
plt.axhline(centralAngle + offset, linestyle="dashdot", color='grey', linewidth=1, label=r"Dwell Limits $\pm 0.5^\circ$")
plt.axhline(centralAngle - offset, linestyle="dashdot", color='grey', linewidth=1)
plt.axhline(centralAngle, linestyle="--", color='grey', linewidth=1, alpha=1/3, label=rf"Central angle $= {centralAngle}^\circ$")
plt.xlabel("Joint A Rotation [Deg]")
plt.ylabel("Joint F Rotation [Deg]")
plt.xlim(0, 200)
plt.ylim(0, 50)
plt.legend(loc='best')
plt.tight_layout()
fig.savefig(os.path.join(HERE, "D8_dwell_zoom.png"))
plt.close(fig)
print(f"\nSaved D8_joint_comparison.png and D8_dwell_zoom.png to {HERE}")
