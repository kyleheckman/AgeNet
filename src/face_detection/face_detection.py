import cv2 as cv
import numpy as np

class Face():
    def __init__(
            self,
            img: np.ndarray,
            index: int
    ):
        self.img = img
        self.index = index
    
    def display(self) -> None:
        cv.imshow(f" Face {self.index}", self.img)

if __name__ == '__main__':
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    stream = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not stream.isOpened:
        print(f"Failed to find stream on ID 0")
        exit()
    
    while (stream.isOpened):
        ret, frame = stream.read()
        if not ret:
            print(f"Failed reading from stream")
            break

        frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(frame_gs, scaleFactor=1.3, minNeighbors=5, minSize=(32,32))
    
        tracking = []
        for index in range(len(faces)):
            x,y,w,h = faces[index]
            tracking.append(Face(frame[y-16 if y >= 16 else 0:y+h+16, x-16 if x >= 16 else 0:x+w+16, :].copy(), index=index))
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        cv.imshow("Video Feed", frame)
        for f in tracking:
            f.display()
        
        if cv.waitKey(1) == ord('q'):
            break

    stream.release()
    cv.destroyAllWindows()