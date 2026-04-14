import cv2

# Haarcascade load
face_cascade = cv2.CascadeClassifier(r"C:\Users\Dell\Desktop\FaceProject\haarcascade_frontalface_default.xml")

# Camera start
cap = cv2.VideoCapture(0)

print("Camera started...")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detect
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Rectangle draw
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Name show (manually)
        name = "Anurag"
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        print(f"{name} Present")

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()