import numpy as np
import cv2 as cv
import dlib

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
    face_classifier = cv.CascadeClassifier('src/pretrained/haarcascade_frontalface_default.xml')
    stream = cv.VideoCapture(0, cv.CAP_DSHOW)
    tracker = dlib.correlation_tracker()

    tracking_face = False

    if not stream.isOpened:
        print(f"Failed to find stream on ID 0")
        exit()
    
    temp_face = None
    while stream.isOpened:
        ret, frame = stream.read()
        if not ret:
            print(f"Failed reading from stream")
            break
        
        if not tracking_face:
            frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(frame_gs, scaleFactor=1.3, minNeighbors=5, minSize=(32,32))

            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0

            for (_x,_y,_w,_h) in faces:
                if _w*_h > maxArea:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)
                    maxArea = w*h

            if maxArea > 0:
                tracker.start_track(
                    frame,
                    dlib.rectangle(x-16,y-32,x+w+16,y+h+16)
                )
                tracking_face = True
                temp_face = Face(frame[y-32:y+h+16, x-16:x+w+16, :].copy(), index=1)
        
        if tracking_face:
            tracking_quality = tracker.update(frame)

            if tracking_quality >= 10:
                tracked_pos = tracker.get_position()

                t_x = int(tracked_pos.left())
                t_y = int(tracked_pos.top())
                t_w = int(tracked_pos.width())
                t_h = int(tracked_pos.height())
                
                cv.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0,0,255), 2)
            else:
                tracking_face = False
        
        cv.imshow("Video Feed", frame)
        if temp_face:
            temp_face.display()
        if cv.waitKey(1) == ord('q'):
            break

stream.release()
cv.destroyAllWindows()