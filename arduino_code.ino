#include <Servo.h>

// Create servo object
Servo handServo;
Servo thumbServo;

// Pin for servo control - using pin 8 and 9 which is PWM capable
const int servoPin = 9;
const int servoPin2 = 8;

// Variable to store the incoming angle
int targetAngle = 90;  // Start at middle position
int targetAngle2 = 90;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Attach servo to its pin
  handServo.attach(servoPin);
  thumbServo.attach(servoPin2);

  // Move to initial position
  handServo.write(targetAngle);
  thumbServo.write(targetAngle2);

}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming angle
  int targetAngle = Serial.parseInt();  // First angle
  int targetAngle2 = Serial.parseInt();  // Second angle

    // Constrain angle to valid range (0-180 degrees)
    targetAngle = constrain(targetAngle, 0, 180);
    targetAngle2 = constrain(targetAngle2, 0, 180);


    // Move servo to target position
    handServo.write(targetAngle);
    thumbServo.write(targetAngle2);
    
    // Clear any remaining data in serial buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}