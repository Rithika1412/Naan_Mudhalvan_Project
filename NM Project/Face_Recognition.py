import cv2
#Load the pre-trained face detection model (Haar cascade)
face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 #Open video stream from webcam (use O for default camera)
cap = cv2.VideoCapture(0)
while True:
    #Read a frame from the webcam
    ret, frame = cap.read() 
    if not ret: 
        break

    # Convert frame to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)

    #Draw bounding boxes around detected faces 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detections 
    cv2.imshow('Facial Recognition - Face Detection', frame)

    #Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release video capture and close window 
cap.release()
cv2.destroyAllWindows()