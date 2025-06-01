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
        self.last_sent_servo_angle2 = -1
        self.last_sent_servo_angle3 = -1
        self.last_sent_servo_angle4 = -1
        self.last_sent_servo_angle5 = -1


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
            return "/dev/ttyACM0"           # Typical default for Linux (could also be '/dev/ttyUSB0')
        elif os_name == "Darwin":
            return "/dev/tty.usbmodem14101"  # Example default for macOS
        else:
            return None                     # unknown system

    def send_to_arduino(self, angle, servo_angle2, servo_angle3, servo_angle4, servo_angle5):
        """
        Send an angle command to the Arduino.

        Args:
            angle (int): Servo angle between 0 and 180 degrees
        """
        # Only send if we're using hardware and the angle has changed
        if self.use_hardware and (angle != self.last_sent_angle or servo_angle2 != self.last_sent_servo_angle2
        or servo_angle3 != self.last_sent_servo_angle3 or servo_angle4 != self.last_sent_servo_angle4
        or servo_angle5 != self.last_sent_servo_angle5):
            try:
                # Convert angle to string and add newline
                command = f"pinky {angle}, ring {servo_angle2}, middle {servo_angle3}, index {servo_angle4}, thumb {servo_angle5}\n"
                self.arduino.write(command.encode())

                # Update last sent angle
                self.last_sent_angle = angle
                self.last_sent_servo_angle2 = servo_angle2
                self.last_sent_servo_angle3 = servo_angle3
                self.last_sent_servo_angle4 = servo_angle4
                self.last_sent_servo_angle5 = servo_angle5
                print(f"Sent angles: {angle}, {servo_angle2}, {servo_angle3}, {servo_angle4}, {servo_angle5}")


                # Optional: Wait for and print Arduino response
                # response = self.arduino.readline().decode().strip()
                # if response:
                #     print(f"Arduino response: {response}")

            except serial.SerialException as e:
                print(f"Failed to send command to Arduino: {e}")
                self.use_hardware = False
                print("Switching to test mode")

    def map_openness_to_gesture(self, pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness):
        """
        Maps the hand openness value to one of three gesture states.

        Args:
            openness (float): Hand openness value between 0 and 1

        Returns:
            tuple: (gesture_name, servo_angle)
        """

        servo_angle = int(pinky_openness * 180)
        servo_angle2 = int(ring_openness * 180)
        servo_angle3 = int(middle_openness * 180)
        servo_angle4 = int(index_openness * 180)
        servo_angle5 = int(thumb_openness * 180)
        total = pinky_openness + ring_openness + middle_openness + index_openness + thumb_openness
        openness_avg = total / 5

        
        if openness_avg < 0.3:
            gesture = "Closed"

        elif openness_avg > 0.7:
             gesture = "Open"

        else:
             gesture = "Middle"

        return gesture, servo_angle, servo_angle2, servo_angle3, servo_angle4, servo_angle5

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

        # for showing fps
        prev_time = 0
        curr_time = 0

        try:
            while True:
                # Read frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read frame")
                    break
                
                # for showing fps
                curr_time = time.time()
                fps = 1/(curr_time - prev_time)
                prev_time = curr_time

                # Process the frame to detect hand and get openness value
                frame, pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness = self.tracker.process_frame(frame)

                # Map hand openness to gesture and servo angle
                gesture, servo_angle, servo_angle2, servo_angle3, servo_angle4, servo_angle5 = self.map_openness_to_gesture(pinky_openness, ring_openness, middle_openness, index_openness, thumb_openness)

                # Send command to Arduino (if using hardware)
                self.send_to_arduino(servo_angle, servo_angle2, servo_angle3, servo_angle4, servo_angle5)

                # Display gesture and angle information on frame
                cv2.putText(frame, f"Gesture: {gesture}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Pinky Servo Angle: {servo_angle}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"ring Servo Angle: {servo_angle2}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"middle Servo Angle: {servo_angle3}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"index Servo Angle: {servo_angle4}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"thumb Servo Angle: {servo_angle5}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(frame, str(int(fps)), (610, 20), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (255, 0, 255), 2)

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
                self.send_to_arduino(180, 180, 180, 180, 180)
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
    system = HandControlSystem(use_hardware=True)
    system.run()

    # Optional: Test with hardware
    # print("\nTesting with Arduino...")
    # system = HandControlSystem(use_hardware=True, com_port='COM3')
    # system.run()


if __name__ == "__main__":
    test_system()