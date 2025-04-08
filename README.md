# Real-Time Gesture Mimicry
 Controlling a 3D-Printed Robotic Hand Using Webcam-Based Hand Tracking

---

# road map
- [x] seperate the thumb from the rest
- [x] translate all openness values to angles
- [x] add depth preception to greatly increase the accuracy (if possible)
- [x] auto detect host os to auto adjust Arduino port path (kinda done, needs testing)
- [x] calculate thumb openness based on its distence from the base of the ring finger?
- [ ] decouple the effects of hand orientation from the openness metric
     - [ ] understand the math
- [ ] make it use the intel iGPU for inferece?
- [ ] extend the code to control each finger seperetly
- [ ] refactor, optimize, etc...
- [ ] rewrite whole project from scratch completely by yourself
