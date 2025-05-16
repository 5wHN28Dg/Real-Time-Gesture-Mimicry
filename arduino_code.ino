#include <Servo.h>

// Create servo object
Servo pinkyServo;
Servo ringServo;
Servo middleServo;
Servo indexServo;
Servo thumbServo;

// Pin for servo control - using pins 9 and onward which are PWM capable
const int servoPin = 9;
const int servoPin2 = 10;
const int servoPin3 = 11;
const int servoPin4 = 12;
const int servoPin5 = 13;

// Variable to store the incoming angle
int targetAngle = 90;  // Start at middle position
int targetAngle2 = 90;
int targetAngle3 = 90;
int targetAngle4 = 90;
int targetAngle5 = 90;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Attach servo to its pin
  pinkyServo.attach(servoPin);
  ringServo.attach(servoPin2);
  middleServo.attach(servoPin3);
  indexServo.attach(servoPin4);
  thumbServo.attach(servoPin5);

  // Move to initial position
  pinkyServo.write(targetAngle);
  ringServo.write(targetAngle2);
  middleServo.write(targetAngle3);
  indexServo.write(targetAngle4);
  thumbServo.write(targetAngle5);

}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming angle
  int targetAngle = Serial.parseInt();  // First angle
  int targetAngle2 = Serial.parseInt();  // Second angle
  int targetAngle3 = Serial.parseInt();
  int targetAngle4 = Serial.parseInt();
  int targetAngle5 = Serial.parseInt();

    // Constrain angle to valid range (0-180 degrees)
    targetAngle = constrain(targetAngle, 0, 180);
    targetAngle2 = constrain(targetAngle2, 0, 180);
    targetAngle3 = constrain(targetAngle3, 0, 180);
    targetAngle4 = constrain(targetAngle4, 0, 180);
    targetAngle5 = constrain(targetAngle5, 0, 180);


    // Move servo to target position
    pinkyServo.write(targetAngle);
    ringServo.write(targetAngle2);
    middleServo.write(targetAngle3);
    indexServo.write(targetAngle4);
    thumbServo.write(targetAngle5);
    delay(20);
    
    // Clear any remaining data in serial buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}