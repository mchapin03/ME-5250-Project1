"""
Runs every analysis script for the one-dwell six-bar, in order, from this folder.
Generates all plots, the CSV and the video.
"""
import os
import subprocess
import sys

FK_INPUT = 0 #crank angle th2 in deg for the FK plot, leave NONE to be asked in terminal
IK_INPUT = 89.3 #target output angle th6 in deg for the IK solve, leave NONE to be asked in terminal

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("six-bar_mechanical_advantage.py", None), #Jacobian, singularities, mechanical advantage (runs first so FK can plot the singularities)
    ("six-bar_one-dwell_closed-form_FK.py", FK_INPUT), #FK, passive joints, joint-range plot
    ("six-bar_one-dwell_IK.py", IK_INPUT), #IK solutions and plots
    ("six-bar_animation_joint_vals_csv.py", None), #CSV, dwell range, video
    ("LivePlotter/data_analysis.py", None), #Physical model error plots
]
for name, answer in SCRIPTS:
    path = os.path.join(HERE, name)
    if not os.path.exists(path): #if failed to find
        print(f"--- Skipping {name}: not found in {HERE}\n")
        continue
    print(f"=== Running {name} ===") #show progress
    stdin_text = None if answer is None else f"{answer}\n"
    if answer is not None:
        print(f"(input {answer} entered automatically)") #if input set in this script
    result = subprocess.run([sys.executable, path], cwd=HERE, input=stdin_text, text=True) #run the script
    if result.returncode != 0: #failure
        sys.exit(f"\n{name} failed (exit code {result.returncode}); stopping.") #failure report
    print(f"=== Finished {name} ===\n") #end script

print("All scripts finished.") #report done