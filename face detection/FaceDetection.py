import cv2
import os

# Create a directory to store the captured face snapshots
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Try to open the camera with different indices (0, 1, 2, etc.)
for camera_index in range(10):
    cam = cv2.VideoCapture(camera_index)
    if cam.isOpened():
        print(f"Using camera with index {camera_index}")
        break
else:
    # If none of the cameras could be opened, print an error message and exit
    print("Error: Could not open any camera.")
    exit(1)

alg = "haarcascade_frontalface_default.xml"  # Accessed the model file
haar_cascade = cv2.CascadeClassifier(alg)  # Loading the model with cv2

snapshot_counter = 0  # Counter for snapshots

while True:
    ret, img = cam.read()  # Read the frame from the camera

    if not ret:
        print("Error: Failed to read a frame.")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting color into grayscale image
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # Get coordinates of faces

    for i, (x, y, w, h) in enumerate(faces):  # Segregating x, y, w, h.
        face_roi = img[y:y + h, x:x + w]  # Crop the face region
        snapshot_counter += 1
        face_filename = os.path.join(output_dir, f"face_{snapshot_counter}.jpg")
        cv2.imwrite(face_filename, face_roi)  # Save the face as a snapshot
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
