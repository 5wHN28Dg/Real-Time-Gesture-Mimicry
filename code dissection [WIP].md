# Dissecting the project to pieces

simple_hand_tracker.py:

* almost all the code is inside a single class 'simplehandtracker' except for:
  * the if statement for running the run_test() function in case the file was run directly
  * the imports

* the simplehandtracker class contains the following functions:
  * __init__:
    * setup mediapipe
  * calculate_hand_openness:
    * get the coordinates of wrist, the base of index and pinky
    * calculate the palm_center from these 3 coordinates
    * get the coordinates of the ring finger base
    * calculate the distance of each finger from the center
    * calculate the distance between the thumb tip and ring finger base
    * get the average 4 finger distance
    * normalize the values

  * process_frame:
        so intuitively speaking, this function does the following:
    * Access the camera
    * start it
    * draw hand landmarks when a hand is detected
    * retrieve hand openness value from the calculate_hand_openness function
    * display the hand and thumb openness value on screen

  * run_test: this function is run only when the simple_hand_tracker is run directly, and it is there to test the functionality of the code, it does the following:
    * start the camera
    * pass the info to the process_frame function to process each frame
    * convert the hand_openness value to a servo angle
    * display the camera live feed with the servo angle overlaid on top of it

hand_control_system.py:
    so we have here 1 class, 6 functions, one if statement, and 6 imports.
    class HandControlSystem:

* __init__ function: initialize the script so that it is ready to run by running the following startup logic:
  * initializes the hand tracker module
  * starts up the process of checking and deciding the appropriate mode to run in, sw or hw mode
  * storage for the last sent arduino angle

* find_arduino_port: look for and connect to the arduino if found through the following:
  * 1st try to connect find and connect to the arduino automatically doing the following:
    * get a list of all the ports
    * look for port with "Arduino" in the description
