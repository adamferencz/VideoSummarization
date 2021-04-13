import sys

import cv2

VIDEO_NAME = 'workout1short.mp4'

cap = cv2.VideoCapture('./in/' + VIDEO_NAME)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
# out = cv2.VideoWriter("./out/graydd_diff.avi", fourcc, 5.0, (1280,720))



ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"DIVX")
out = cv2.VideoWriter("./out/gray.mp4", fourcc, fps, (1280, 720))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
curr_frame = 0
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue


    image = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (1280, 720))
    out.write(image)
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret:
        break

    sys.stdout.write("\r Frame %s/%s " % (curr_frame, frame_count))
    sys.stdout.flush()

    curr_frame += 1

cap.release()
out.release()
