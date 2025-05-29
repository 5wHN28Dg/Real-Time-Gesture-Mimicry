# Real-Time Gesture Mimicry

*Controlling a 3D-Printed Robotic Hand Using Webcam-Based Hand Tracking*

---

### 🚀 Project Overview

This repo doubles as my graduation project and personal learning journey. My end goal is to rewrite the entire codebase from scratch. I honestly wasn’t planning to make it public until then, but I’m curious if opening it up early might help me (and maybe others) in any way so let’s see how this goes!

---

#### ⚠️ Disclaimer

This code started as a template from Claude AI, and I’ve since heavily hacked on it—I'm a total beginner in Python (and coding in general), so still learning as I go.

---

### 🔧 Dependencies

* **mediapipe** (hand recognition and tracking)
* **OpenCV** (video capture and image processing)
* **pySerial** (Arduino detection and communication)

---

### 🗂 File Structure

* `simple_hand_tracker.py` — starts the camera, detects your hand, calculates openness metrics.
* `hand_control_system.py` — converts openness values to servo angles and sends commands to the Arduino.
* `Arduino_code.ino` — sets up the Arduino to receive angles and drive the servos.
* `code dissection (WIP).md` — in-depth breakdown of how everything works.
* `requirements.txt` — all project dependencies.

---

### Control Logic flowchart

<p align="center">
  <img src="Assets/flowchart.svg" alt="Animated SVG" width="600" />
</p>

### 🔍 Quick Tips

* **Latest changes?** Check the `testing` branch for the freshest updates.
* **Code breakdown?** Peek at [code dissection.md (WIP)](https://github.com/5wHN28Dg/Real-Time-Gesture-Mimicry/blob/main/code%20dissection%20%5BWIP%5D.md).

---

### 📝 TODO

* [x] Separate thumb from other fingers.
* [x] Map openness values to servo angles.
* [x] Add depth perception to improve accuracy (if possible).
* [x] make the openness metric more invariant to hand size and distance from camera
* [x] Auto-detect host OS and adjust Arduino port path (needs more testing).
* [x] Calibrate thumb openness relative to the ring-finger base.
* [ ] Decouple hand orientation from openness metric:

  * [ ] Grasp the underlying math.
* [ ] Leverage Intel iGPU for inference.
* [ ] Write comprehensive documentation ⏸️
  * [x] phase 1: initial draft
  * [ ] phase 2: a deep dive into the technical and mathematical details
  * [ ] phase 3: polishing
* [ ] Self-Adjusting Calibration process so that the system self-adjusts to different users’ hand sizes without manual baseline tweaking
  * [ ] Initial Calibration Routine.
  * [ ] Running Min/Max Tracking.
  * [ ] Statistical & ML-Based Normalization.
* [x] Extend control to each finger individually.
  * [x] initial implementation
  * [x] review
  * [x] hardware testing
* [ ] Refactor and optimize.
* [ ] Rewrite the entire project from scratch ⏸️

---

### 🤝 Contributions & Feedback

Feel free to submit issues, suggest features, or even open a PR—your insights could speed up my learning and improve the project for everyone!
