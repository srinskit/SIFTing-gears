from PIL import ImageGrab
import numpy as np
import cv2
import keyboard
from time import sleep
from simple_pid import PID
map_patch = np.zeros((250, 250, 3))

speed_patch = np.zeros((100, 640, 3))
template_sift = cv2.ORB_create(nfeatures=30)
frame_sift = cv2.ORB_create(nfeatures=100)
template = cv2.imread('./CV_DS/t24.png', cv2.IMREAD_COLOR)
template = cv2.GaussianBlur(template, (17, 17), 0)
kp1, des1 = template_sift.detectAndCompute(template, None)
out = None
key = "w"
frame_size = (1280, 342)
control = 0
fps = 30


quit = True
def toggle_quit(event):
    global quit
    quit = not quit

def dist_point(line, point):
    a, b, c = line
    x, y = tuple(point)
    return abs(a * x + b * y + c) / (a ** 2 + b ** 2) ** 0.5

def consider_line(line, points, dist_threshold, count_threshold):
    count_inliers = 0
    for point in points:
        if dist_point(line, point) <= dist_threshold:
            count_inliers += 1
    if count_inliers >= count_threshold * len(points):
        return True
    else:
        return False


def get_err(frame):
    global template_sift, template
    pad_time = 1/fps
    frame[-250:, :250] = map_patch
    frame[-100:, -640:] = speed_patch
    frame = frame[int(frame.shape[0] / 1.8):]
    h, w, _ = frame.shape
    frame = cv2.GaussianBlur(frame, (17, 17), 0)
    kp2, des2 = frame_sift.detectAndCompute(frame, None)
    template = cv2.drawKeypoints(template, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=max(1, int(.03 * len(des2))))
    points = [[], []]
    key_points = []
    distances = []
    for k_neighbours in matches:
        for match in k_neighbours:
            distances.append(match.distance)
            points[0].append((int(kp2[match.trainIdx].pt[0])))
            points[1].append((int(kp2[match.trainIdx].pt[1])))
            key_points.append((int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])))
    for p in key_points:
        cv2.circle(frame, p, 10, (0, 0, 255), -1)
    [vx, vy, x, y] = cv2.fitLine(np.array(key_points), cv2.DIST_WELSCH, 0, 0.01, 0.01)
    m = 100
    x0, y0 = int(x[0] - m * vx[0]), int(y[0] - m * vy[0])
    x1, y1 = int(x[0] + m * vx[0]), int(y[0] + m * vy[0])
    a = vy[0] / vx[0]
    b = -1
    c = y0 - a * x0
    frame = cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 5)
    if not consider_line((a, b, c), key_points, 50, .80):
        # cv2.putText(frame,'key: %s'%key,(10,h-75), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        # cv2.imshow("win", frame)
        # cv2.waitKey(0)
        # out.write(frame)
        # sleep(pad_time)
        return None
    err_y = h // 2
    ideal_dist = 675
    if y0 != y1:
        err_x = x1+(err_y-y1)*(x0-x1)/(y0-y1)
    else:
        err_x = x0
    err = err_x - ideal_dist
    sleep(pad_time)
    cv2.putText(frame,'key: %s'%key,(10,h-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,'ctr: %d'%control,(10,h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,'err: %d'%err,(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    out.write(frame)
    cv2.imshow("win", frame)
    cv2.waitKey(0)
    return err


def update(control, err):
    pass

def main():  
    global quit, out, control
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('.\CV_DS\debug_1.avi', fourcc, fps, frame_size)
    keyboard.on_press_key("1", toggle_quit, suppress=True)
    pid = PID(1, 0.1, 0.05, setpoint=1)
    pid.sample_time = 1
    print("Press o to start")
    while quit:
        sleep(1)
    print("Press o to quit")        
    control = 0
    err = 0
    while not quit:
        frame = np.array(ImageGrab.grab())
        new_err = get_err(frame)
        if new_err is not None:
            err = new_err
        control = pid(err)
        update(control, err)
    cv2.destroyAllWindows()
    keyboard.unhook_all()
    out.release()

def main1():  
    global quit, control, out
    vid = cv2.VideoCapture("./CV_DS/7.mp4")
    success = 1
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('.\CV_DS\debug_1.avi', fourcc, fps, frame_size)
    while success:
        success, frame = vid.read()
        if not success:
            break
        get_err(frame)
        ch = cv2.waitKey(10)
        if ch == ord('q'):
            break
    out.release()
    



if __name__ == '__main__':
    main1()