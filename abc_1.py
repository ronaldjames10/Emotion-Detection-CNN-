import cv2

# List of possible backends
backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW]

camera = None

for backend in backends:
    for index in range(5):  # try indexes 0-4
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"✅ Camera found at index {index} with backend {backend}")
            camera = cap
            break
    if camera:
        break

if not camera:
    print("❌ No accessible camera found")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

camera.release()
cv2.destroyAllWindows()
