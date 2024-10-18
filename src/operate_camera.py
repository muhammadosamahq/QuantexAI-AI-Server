import cv2

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 's' to capture and save the image, or 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Display the live video feed
    cv2.imshow("Webcam - Press 's' to capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Save the captured frame as an image file
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved as 'captured_image.jpg'")
    elif key == ord("q"):
        # Quit the program
        print("Exiting...")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
