import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    finger_counts = []

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        total_fingers = 0
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    finger_tips = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        if d > 10000:  # Filter based on the distance
                            if start[1] < frame.shape[0] / 2:  # Filter to ignore the lower half of the frame
                                finger_tips.append(start)
                                cv2.circle(frame, start, 8, (0, 0, 255), -1)
                                cv2.putText(frame, f"Finger {len(finger_tips)}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    total_fingers = len(finger_tips)
        cv2.putText(frame, f"Total fingers: {total_fingers}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
