# ME-5250 Robot Mechanics & Control Project 1
## Six-bar One-Dwell Mechanism

### How to Run
Install the dependencies:

```
pip install -r requirements.txt
```

Then, from the main project folder, run every analysis script and regenerate all plots and data:

```
python run_all.py
```

(Use `python3` on macOS/Linux if `python` points to Python 2.) This can also be run from inside your IDE.
Keep the repository's folder structure as is: `data_analysis.py` and `data.txt` must stay in `LivePlotter/`.

The FK and IK scripts each ask for an input angle. To run everything without prompts, set
`FK_INPUT` and `IK_INPUT` at the top of `run_all.py`.

### Running scripts individually
Each script can also be run on its own, but some use files created by others:

| Script | Needs | Creates |
|---|---|---|
| `six-bar_one-dwell_D6.py` | — | `D6_mechanical_advantage.png`, `D6_singularities.csv`, `D5_values.csv` |
| `six-bar_one-dwell_closed-form_FK.py` | `D6_singularities.csv` (optional, for the singularity markers) | `fk_sixbar.png`, `fk_sixbar_ranges.png` |
| `six-bar_one-dwell_IK.py` | — | `ik_sixbar.png` |
| `six-bar_animation_joint_vals_csv.py` | — | `sixbar_joint_angles.csv`, `sixbar_rotation.mp4` |
| `LivePlotter/data_analysis.py` | `sixbar_joint_angles.csv`, `LivePlotter/data.txt` | `D8_joint_comparison.png`, `D8_dwell_zoom.png` (these generate inside the 'LivePlotter' subfolder) |

`run_all.py` runs them in this order, so every dependency is created before it is needed.

### Live data capture (hardware only)
`LivePlotter/plotter.py` reads joint angles from the ESP32 over serial and logs them to
`LivePlotter/data.txt`. It requires the physical model connected over USB and is not run by
`run_all.py`.
