# Real-Time Gesture Mimicry

**Controlling a 3D-Printed Robotic Hand Using Webcam-Based Hand Tracking**


## Overview

This project mirrors a human hand in real time using:

- A webcam for hand tracking
  
- MediaPipe for landmark detection
  
- An Arduino-controlled 3D-printed robotic hand
  

The system detects finger openness and maps it to servo angles, allowing the robotic hand to mimic live gestures.

This repository documents the full pipeline: vision → metric extraction → control logic → actuation.


## Demo

<p align="center"> <a href="https://www.youtube.com/watch?v=6zBArJ0yIYY"> <img src="https://img.youtube.com/vi/6zBArJ0yIYY/maxresdefault.jpg" alt="Watch on YouTube" width="600" /> </a> </p>


## System Architecture

1. **Hand Detection**
  
  - MediaPipe extracts 21 hand landmarks.
2. **Openness Metric**
  
  - Finger states are calculated relative to anatomical reference points.
3. **Mapping Layer**
  
  - Openness values are converted into servo angles.
4. **Actuation**
  
  - Angles are transmitted to the Arduino via serial communication.
5. **Execution**
  
  - Servos drive the 3D-printed hand accordingly.

### Control Flow

<p align="center"> <img src="Assets/flowchart.svg" alt="Animated SVG" width="600" /> </p>

## Project Status

> ⚠ Development is currently paused.

The system works in its current experimental form, but it is not production-ready.  
A full rewrite is planned in the future to achieve architectural clarity and conceptual ownership.


## Dependencies

- `mediapipe` — hand landmark detection
  
- `opencv-python` — video capture and processing
  
- `pyserial` — Arduino communication
  

Install with:

```bash
pip install -r requirements.txt
```


## File Structure

- `simple_hand_tracker.py` — camera + landmark detection + openness calculation
  
- `hand_control_system.py` — mapping + serial communication
  
- `Arduino_code.ino` — servo control firmware
  
- `code dissection [WIP].md` — technical breakdown
  
- `requirements.txt` — Python dependencies
  


## Philosophy

This project is built primarily for my own use and learning.

It evolves when I need something, want to improve something, or decide to explore an idea.  
It is not driven by feature requests, deadlines, or a public roadmap.

You are welcome to use it, fork it, or build on it.

However:

- There are no guarantees of support
  
- There are no timelines
  
- Feature requests may be ignored
  
- The direction of the project is decided solely by me
  

If you need a specific feature, open a PR and implement it.

Donations are appreciated, but they do not grant influence over the project.

This is a personal tool first, public software second.


## Roadmap (Long-Term)

- Decouple hand orientation from openness metric
  
- Improve calibration robustness
  
- Hardware abstraction for variable actuator counts
  
- Packaging for Linux / macOS / Windows
  
- Full rewrite with redesigned architecture
  


## Contributions

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.
