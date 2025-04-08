# Real-Time Gesture Mimicry
 Controlling a 3D-Printed Robotic Hand Using Webcam-Based Hand Tracking

### disclaimer: This code is based on a code template generated by Claude AI, I am a complete beginner in python (and coding in general) and still learning and the code as it is right now (after making many improvements and changes) is a combination of AI generated code + Stack Overflow + me.

This project is a part of my learning journey, so the end goal is to be able to rewrite the whole code from scratch all by myself, and honestly I wasn't planning on making this repo public until I rewrite the whole project from scratch but hmm... I wanna experiment and see if making it public now will help me in any way and I guess it could help others too so let's see how this will go!

This project uses the following for hand recognition and tracking:
- mediapipe
- openCV

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
- [ ] rewrite whole project from scratch all by myself
