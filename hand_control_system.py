import cv2
import serial
import time
from simple_hand_tracker import SimpleHandTracker
import platform
from serial.tools import list_ports

class HandControlSystem:
    def __init__(self, use_hardware=True, com_port=None):
        """
        Initialize the hand control system.

        Args:
            use_hardware (bool): If True, connects to Arduino. If False, runs in test mode.
            com_port (str): The COM port where Arduino is connected (e.g., 'COM3' on Windows
                          or '/dev/ttyUSB0' on Linux)
        """
        # Initialize the hand tracker
        self.tracker = SimpleHandTracker()

        # if no port is provided, detect automatically
        if com_port is None:
            com_port = self.find_arduino_port()
            print(f"Attempting connection on port: {com_port}")

        # Flag to track if we're using real hardware
        self.use_hardware = use_hardware

        # Variable to store the last sent angle (to avoid unnecessary serial communication)
        self.last_sent_angle = -1
        self.last_sent_servo2_angle = -1


    # Connect to Arduino if we're using hardware
        if use_hardware:
            try:
                # Try to establish serial connection
                self.arduino = serial.Serial(com_port, 9600, timeout=1)
                print(f"Connected to Arduino on {com_port}")

                # Wait for Arduino to initialize
                time.sleep(2)
                print("Arduino connection initialized")

            except serial.SerialException as e:
                print(f"Failed to connect to Arduino: {e}")
                print("Running in test mode instead")
                self.use_hardware = False

    def find_arduino_port(self):
        """
        Scans available serial ports and returns the port that likely belongs to an Arduino.
        Falls back to a default port based on the host operating system if no Arduino is found.
        """
        # get all available ports
        ports = list(list_ports.comports())

        # look for a port with 'Arduino' in its description
        for port in ports:
            if "Arduino" in port.description:
                return port.device
        
        # fallback defaults based on OS
        os_name = platform.system()
        if os_name == "Windows":
            return "COM3"                   # default for windows
        elif os_name == "Linux":
            return "/dev/ttyUSB0"           # Typical default for Linux (could also be '/dev/ttyACM0')
        elif os_name == "Darwin":
            return "dev/tty.usbmodem14101"  # Example default for macOS
        else:
            return None                     # unknown system

    def send_to_arduino(self, angle, servo2_angle):
        """
        Send an angle command to the Arduino.

        Args:
            angle (int): Servo angle between 0 and 180 degrees
        """
        # Only send if we're using hardware and the angle has changed
        if self.use_hardware and (angle != self.last_sent_angle or servo2_angle != self.last_sent_servo2_angle):
            try:
                # Convert angle to string and add newline
                command = f"{angle}, {servo2_angle}\n"
                self.arduino.write(command.encode())

                # Update last sent angle
                self.last_sent_angle = angle
                self.last_sent_servo2_angle = servo2_angle
                print(f"Sent angle: {angle}")
                print(f"Servo2 Angle: {servo2_angle}")

                # Optional: Wait for and print Arduino response
                # response = self.arduino.readline().decode().strip()
                # if response:
                #     print(f"Arduino response: {response}")

            except serial.SerialException as e:
                print(f"Failed to send command to Arduino: {e}")
                self.use_hardware = False
                print("Switching to test mode")

    def map_openness_to_gesture(self, openness, thumb_openness):
        """
        Maps the hand openness value to one of three gesture states.

        Args:
            openness (float): Hand openness value between 0 and 1

        Returns:
            tuple: (gesture_name, servo_angle)
        """

        servo_angle = int(openness * 180)
        servo2_angle = int(thumb_openness * 180)
        
        if openness < 0.3:
            gesture = "Closed"

        elif openness > 0.7:
             gesture = "Open"

        else:
             gesture = "Middle"

        return gesture, servo_angle, servo2_angle

    def run(self):
        """
        Main loop of the system. Captures video, processes hand gestures,
        and controls the servo.
        """
        print("Starting hand control system...")
        print("Press 'q' to quit")

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        try:
            while True:
                # Read frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read frame")
                    break

                # Process the frame to detect hand and get openness value
                frame, hand_openness, thumb_openness = self.tracker.process_frame(frame)

                # Map hand openness to gesture and servo angle
                gesture, servo_angle, servo2_angle = self.map_openness_to_gesture(hand_openness, thumb_openness)

                # Send command to Arduino (if using hardware)
                self.send_to_arduino(servo_angle, servo2_angle)

                # Display gesture and angle information on frame
                cv2.putText(frame, f"Gesture: {gesture}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.putText(frame, f"Servo Angle: {servo_angle}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.putText(frame, f"Servo2 Angle: {servo2_angle}",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                # Show the frame
                cv2.imshow('Hand Control System', frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Clean up resources
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()

            if self.use_hardware:
                # Move servo to neutral position before closing
                self.send_to_arduino(90, 90)
                time.sleep(0.5)  # Wait for servo to move
                self.arduino.close()
                print("Arduino connection closed")


def test_system():
    """
    Test function to verify system components.
    """
    print("Testing hand control system...")

    # Test without hardware first
    print("\nTesting without Arduino (test mode)...")
    system = HandControlSystem(use_hardware=False)
    system.run()

    # Optional: Test with hardware
    # print("\nTesting with Arduino...")
    # system = HandControlSystem(use_hardware=True, com_port='COM3')
    # system.run()


if __name__ == "__main__":
    test_system()