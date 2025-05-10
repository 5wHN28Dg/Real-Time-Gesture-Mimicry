# Dissecting the project to pieces

## simple_hand_tracker.py

- almost all the code is inside a single class 'simplehandtracker' except for:

  - the if statement for running the run_test() function in case the file was run directly
  - the imports

  - the simplehandtracker class contains the following functions:

    - **init**:
      - setup mediapipe
    - calculate_hand_openness:

      - get the coordinates of wrist, the base of index and pinky
      - calculate the palm_center from these 3 coordinates
      - get the coordinates of the ring finger base
      - calculate the distance of each finger from the center
      - calculate the distance between the thumb tip and ring finger base
      - get the average 4 finger distance
      - normalize the values

    - process_frame:
      so intuitively speaking, this function does the following:

      - Access the camera
      - start it
      - draw hand landmarks when a hand is detected
      - retrieve hand openness value from the calculate_hand_openness function
      - display the hand and thumb openness value on screen

    - run_test: this function is run only when the simple_hand_tracker is run directly, and it is there to test the functionality of the code, it does the following:
      - start the camera
      - pass the info to the process_frame function to process each frame
      - convert the hand_openness value to a servo angle
      - display the camera live feed with the servo angle overlaid on top of it

## hand_control_system.py

  so we have here 1 class, 6 functions, one if statement, and 6 imports.
  class HandControlSystem:

- **init** function: initialize the script so that it is ready to run by running the following startup logic:

  - initializes the hand tracker module
  - starts up the process of checking and deciding the appropriate mode to run in, sw or hw mode
  - storage for the last sent arduino angle

- find_arduino_port: look for and connect to the arduino if found through the following:

  - 1st try to find and connect to the arduino automatically doing the following:
    - get a list of all the ports
    - look for a port with "Arduino" in the description
  - if not then fallback to OS based defaults

- send_to_arduino: sends a angle command to the arduino by starting an if statement that only runs if use_hardware and one the current angle is different than the last one, the if statement contains the following:

  - and handle errors gracefully a try except is used so try to:
    - convert the angle to a string
    - send it the arduino
    - update the last sent angle
    - print sent angles and if you fail to send the angle then:
      - print a message telling the user that you failed
      - switch back to the sw/testing mode

- map_openness_to_gesture: the mapping process is pretty simple, since the openness value is 0 - 1 then you just have to multiply it by 180 to get the matching servo angle and that's pretty much what I did then I used an if elif else statement to assign a keyword to the gesture variable based on the openness value.

- run: just as the comment in the code says so:
  - assign a variable to the camera video stream
  - if the camera could't be started then throw an error message then exit if not then continue

  - try to run the following while loop
    - read from camera
      - if failed print an error message then exit.
    - send the camera stream 'frame' to the process_frame function in the simple_hand_tracker aka tracker to get back the frame, hand openness, thumb openness
    - send the openness value to the map_openness_to_gesture function to get the servo angle and gesture
    - call the send_to_arduino function to provide it with the servo angle so that it sends it to the arduino
    - display the current angle, gesture on the screen/camera frame
    - show the camera stream with the text
    - set the key 'q' as the keyboard key to close the program
    - do some housekeeping right before you close the program:
      - free the camera
      - nuke any remaining window
      - if the program is run in hw mode then set the servos to 90 then wait for the servos to move then close the serial connection with the arduino

- test_system: this function will run if the file was run directly, it job is to either run the program in hw or sw mode then start the run function

- if **name** == "**main**": just an if statement that will call the test_system function only if the program was run directly, not called or imported from within another file.

## arduino_code.ino

- outside the 2 functions
  - assign a virtual avatar for each servo that exist in the real world, what we are doing here is representing the actual servo motors we have in sw, we are saying create 2 object of the type servo, one named handServo, the other is thumbServo
  - then we are just creating 2 constants that are integers and constant, just to store the pin numbers that we gonna connect the servos's control wire to
  - create two variables to store the incoming angles in and set an optional and randomly chosen initial angle for the 2 servos

- void setup: setup some shit we gonna need
  - initialize serial communication, setting the baud rate at 9600
  - attach each servo to on of the 2 pins we defined earlier
  - send the initial target servo angle to each servo (optional, not really necessary)

- void loop: in here we gonna put the code that we want to be kept running
  - 1st check to see if you can read any serial data of the USB connection, if yes then:
    - read the incoming data, since we sat it up to be coming in pairs of two, save the 1st one as the new targetAngle and the 2nd as the 2nd...
    - now we gonna use the constrain function which will act as guardrails or boundaries for the received angle, it will return the angle value to the next closest within range angle value if the received angle is out of the permitted range (if it is 190 then it will change it to 180...)
    - now we gonna send the new angle value to the servos
    - just some house keeping, making sure the buffer is dead empty before the loop starts all over again
