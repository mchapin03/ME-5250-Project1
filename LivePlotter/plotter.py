#!/usr/bin/env python3
"""
Live-plot Joint D and Joint F against Joint A from an ESP32 over serial.

Expected line format (one reading per line, newline-terminated):
    Joint A: 13952 | Joint D: -1469 | Joint F: 383

Joints are matched by name, so their order doesn't matter and extra joints
in the line are ignored.

Install:
    pip install pyserial pyqtgraph PyQt6

Usage:
    python esp32_joint_plotter.py                     # auto-picks the port if only one is connected
    python esp32_joint_plotter.py --port COM5
    python esp32_joint_plotter.py --port COM5 --window 5
"""

import argparse
import queue
import re
import sys
import threading
import time
from collections import deque

import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore
from serial.tools import list_ports

# Matches every "Joint <name>: <int>" pair in a line (bytes, no decoding needed)
JOINT_RE = re.compile(rb"Joint\s+(\w+):\s*(-?\d+)")
REQUIRED = (b"A", b"D", b"F")
DEG_PER_COUNT = 360 / 4095  # sensor outputs 0-4095


def parse_args():
    p = argparse.ArgumentParser(description="Live plot ESP32 Joint D and F vs Joint A")
    p.add_argument("--port", help="Serial port (e.g. COM5, /dev/ttyUSB0, /dev/cu.usbserial-XXXX)")
    p.add_argument("--baud", type=int, default=500000, help="Baud rate (default: 500000)")
    p.add_argument("--window", type=float, default=10.0,
                   help="Seconds of history kept in the trace (default: 10)")
    return p.parse_args()


def pick_port(requested):
    """Use the requested port, or auto-select if exactly one port is available."""
    if requested:
        return requested
    ports = list(list_ports.comports())
    if len(ports) == 1:
        print(f"Using {ports[0].device} ({ports[0].description})")
        return ports[0].device
    print("Specify a port with --port. Available ports:")
    if not ports:
        print("  (none found)")
    for port in ports:
        print(f"  {port.device}  {port.description}")
    sys.exit(1)


def serial_reader(ser, out_q, stop_event):
    """Background thread: read serial in bulk, parse lines, push batches of (t, a, d, f)."""
    t0 = time.monotonic()
    pending = b""
    ser.reset_input_buffer()

    while not stop_event.is_set():
        try:
            # Grab everything waiting in the OS buffer in one call
            chunk = ser.read(ser.in_waiting or 1)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            break
        if not chunk:
            continue

        pending += chunk
        *lines, pending = pending.split(b"\n")
        if len(pending) > 4096:  # no newline for a long time (wrong baud?) -> discard
            pending = b""

        now = time.monotonic() - t0
        batch = []
        for line in lines:
            joints = dict(JOINT_RE.findall(line))
            if all(name in joints for name in REQUIRED):
                batch.append((now,
                              int(joints[b"A"]) * DEG_PER_COUNT,
                              int(joints[b"D"]) * DEG_PER_COUNT,
                              int(joints[b"F"]) * DEG_PER_COUNT))
        if batch:
            out_q.put(batch)


def main():
    args = parse_args()
    port = pick_port(args.port)

    try:
        ser = serial.Serial(port, args.baud, timeout=0.05)
    except serial.SerialException as e:
        sys.exit(f"Could not open {port}: {e}")

    data_q = queue.Queue()
    stop_event = threading.Event()
    reader = threading.Thread(target=serial_reader, args=(ser, data_q, stop_event), daemon=True)
    reader.start()

    # ---------------- GUI setup ----------------
    app = pg.mkQApp("ESP32 Joint Plotter")
    pg.setConfigOptions(antialias=True)  # antialiasing is a large render cost

    win = pg.GraphicsLayoutWidget(title=f"ESP32 Joints - {port} @ {args.baud}")
    win.resize(1000, 700)
    win.show()

    plot_d = win.addPlot(row=0, col=0, title="Joint D vs Joint A")
    plot_f = win.addPlot(row=1, col=0, title="Joint F vs Joint A")
    plot_f.setXLink(plot_d)  # both plots share the Joint A axis

    for plot, name in ((plot_d, "Joint D [degrees]"), (plot_f, "Joint F [degrees]")):
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("left", name)
        plot.setLabel("bottom", "Joint A [degrees]")
        # Note: no clip-to-view or downsampling here; both assume x is sorted,
        # which isn't true when x is a joint that moves back and forth.

    # Trace of recent history, plus a dot marking the current position
    trace_d = plot_d.plot(pen=pg.mkPen("#4fc3f7", width=1), skipFiniteCheck=True)
    trace_f = plot_f.plot(pen=pg.mkPen("#ff8a65", width=1), skipFiniteCheck=True)
    dot_d = plot_d.plot(pen=None, symbol="o", symbolSize=9, symbolBrush="#ffffff", symbolPen=None)
    dot_f = plot_f.plot(pen=None, symbol="o", symbolSize=9, symbolBrush="#ffffff", symbolPen=None)

    t_buf, a_buf, d_buf, f_buf = deque(), deque(), deque(), deque()

    def update():
        got_data = False
        # Drain every batch the reader thread has queued since the last frame
        while True:
            try:
                batch = data_q.get_nowait()
            except queue.Empty:
                break
            ts, a_vals, d_vals, f_vals = zip(*batch)
            t_buf.extend(ts)
            a_buf.extend(a_vals)
            d_buf.extend(d_vals)
            f_buf.extend(f_vals)
            got_data = True

        if not got_data:
            return

        # Keep only the last `window` seconds of samples
        cutoff = t_buf[-1] - args.window
        while t_buf and t_buf[0] < cutoff:
            t_buf.popleft()
            a_buf.popleft()
            d_buf.popleft()
            f_buf.popleft()

        a_list = list(a_buf)
        trace_d.setData(a_list, list(d_buf))
        trace_f.setData(a_list, list(f_buf))

        a, d, f = a_buf[-1], d_buf[-1], f_buf[-1]
        dot_d.setData([a], [d])
        dot_f.setData([a], [f])
        plot_d.setTitle(f"Joint D vs Joint A    (A: {a}, D: {d})")
        plot_f.setTitle(f"Joint F vs Joint A    (A: {a}, F: {f})")
        with open("data.txt", "a") as file:
            file.write(f"A: {a}, D: {d}, F: {f}\n")
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(8)

    try:
        pg.exec()
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        ser.close()


if __name__ == "__main__":
    main()