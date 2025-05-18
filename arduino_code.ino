#include <Servo.h>

// Create servo object
Servo pinkyServo;
Servo ringServo;
Servo middleServo;
Servo indexServo;
Servo thumbServo;

// Pin for servo control - using the pins that are PWM capable
const int servoPin = 3;   // pinky
const int servoPin2 = 5;  // ring
const int servoPin3 = 6;  // middle
const int servoPin4 = 9;  // index
const int servoPin5 = 10; // thumb

// Variables to store the incoming angles
int pinkyAngle = 0;
int ringAngle = 0;
int middleAngle = 0;
int indexAngle = 0;
int thumbAngle = 0;

// Buffer to store incoming serial data
String inputString = "";      // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Reserve 200 bytes for the inputString
  inputString.reserve(200);
  
  // Attach servo to its pin
  pinkyServo.attach(servoPin);
  ringServo.attach(servoPin2);
  middleServo.attach(servoPin3);
  indexServo.attach(servoPin4);
  thumbServo.attach(servoPin5);
  
  // Move to initial position
  pinkyServo.write(pinkyAngle);
  ringServo.write(ringAngle);
  middleServo.write(middleAngle);
  indexServo.write(indexAngle);
  thumbServo.write(thumbAngle);
}

void loop() {
  // When a complete command string is received
  if (stringComplete) {
    // Parse the command string
    parseCommand(inputString);
    
    // Clear the string for the next command
    inputString = "";
    stringComplete = false;
  }
}

// This function is called whenever serial data is available
void serialEvent() {
  while (Serial.available()) {
    // Get the new byte
    char inChar = (char)Serial.read();
    
    // Add it to the inputString
    inputString += inChar;
    
    // If the incoming character is a newline, set a flag so the main loop can
    // process the string
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}

// Parse the command string and extract servo angles
void parseCommand(String command) {
  // Variables to store the temporary angles
  int angle = 0;
  
  // Convert command to lowercase for easier parsing
  command.toLowerCase();
  
  // Look for each finger name and extract its angle
  
  // Pinky finger
  int pinkyIndex = command.indexOf("pinky");
  if (pinkyIndex != -1) {
    // Find the number after "pinky"
    angle = extractAngle(command, pinkyIndex + 5);
    if (angle >= 0) {
      // Constrain to valid range
      pinkyAngle = constrain(angle, 0, 180);
      pinkyServo.write(pinkyAngle);
    }
  }
  
  // Ring finger
  int ringIndex = command.indexOf("ring");
  if (ringIndex != -1) {
    // Find the number after "ring"
    angle = extractAngle(command, ringIndex + 4);
    if (angle >= 0) {
      // Constrain to valid range
      ringAngle = constrain(angle, 0, 180);
      ringServo.write(ringAngle);
    }
  }
  
// Middle finger
int middleIndex = command.indexOf("middle");
if (middleIndex != -1) {
  // Find the number after "middle"
  angle = extractAngle(command, middleIndex + 6);
  if (angle >= 0) {
    // Constrain to valid range
    middleAngle = constrain(angle, 0, 180);
    // Reverse the angle (180 - angle)
    middleAngle = 180 - middleAngle;
    middleServo.write(middleAngle);
  }
}

// Index finger
int indexIndex = command.indexOf("index");
if (indexIndex != -1) {
  // Find the number after "index"
  angle = extractAngle(command, indexIndex + 5);
  if (angle >= 0) {
    // Constrain to valid range
    indexAngle = constrain(angle, 0, 180);
    // Reverse the angle (180 - angle)
    indexAngle = 180 - indexAngle;
    indexServo.write(indexAngle);
  }
}
  
  // Thumb
  int thumbIndex = command.indexOf("thumb");
  if (thumbIndex != -1) {
    // Find the number after "thumb"
    angle = extractAngle(command, thumbIndex + 5);
    if (angle >= 0) {
      // Constrain to valid range
      thumbAngle = constrain(angle, 0, 180);
      thumbServo.write(thumbAngle);
    }
  }
  
  // Small delay to allow servos to move
  delay(20);
}

// Helper function to extract an angle from a substring
int extractAngle(String command, int startIndex) {
  // Skip any spaces or non-digit characters
  while (startIndex < command.length() && !isDigit(command.charAt(startIndex))) {
    startIndex++;
  }
  
  // If we reached the end of the string, return error
  if (startIndex >= command.length()) {
    return -1;
  }
  
  // Extract digits until we find a non-digit character
  String angleStr = "";
  while (startIndex < command.length() && (isDigit(command.charAt(startIndex)))) {
    angleStr += command.charAt(startIndex);
    startIndex++;
  }
  
  // Convert to integer and return
  return angleStr.toInt();
}