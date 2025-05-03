Get the SCPs for each finger
get the average of each one from the wrist
calculate the distance from the tip to the average
translate the openness value for each finger to servo motor angle
add 3 more servo motors
display the openness value for each finger

display the openness value for each finger

------------------------------------

Dissecting the project to pieces
let's start with simple_hand_tracker.py:

* almost all the code is inside a single class 'simplehandtracker' except for:
    * the if statement for running the run_test() function in case the file was run directly
    * the imports

* the simplehandtracker class contains the following functions:
    * __init__:
        * setup mediapipe
    * calculate_hand_opennness:
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

now for the hand_control_system.py:
    so we have here 1 class, 6 functions and one if statement and 6 imports
    class HandControlSystem:
    