#include <Servo.h>

// Create servo object
Servo handServo;

// Pin for servo control - using pin 9 which is PWM capable
const int servoPin = 9;

// Variable to store the incoming angle
int targetAngle = 90;  // Start at middle position

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo to its pin
  handServo.attach(servoPin);
  
  // Move to initial position
  handServo.write(targetAngle);
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming angle
    targetAngle = Serial.parseInt();
    
    // Constrain angle to valid range (0-180 degrees)
    targetAngle = constrain(targetAngle, 0, 180);
    
    // Move servo to target position
    handServo.write(targetAngle);
    
    // Clear any remaining data in serial buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}