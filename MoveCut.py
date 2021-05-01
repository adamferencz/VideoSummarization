# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
import datetime
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time




def find_countours(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_importance(video_name):
    cap = cv2.VideoCapture('./in/'+video_name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    curr_frame = 0

    fps = cap.get((cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    big_contours_counts = []
    all_contours_counts = []
    big_contours_sizes = []
    all_contours_sizes = []
    big_contours_mul = []
    all_contours_mul = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")#DIVX
    output = cv2.VideoWriter("./out/" + "diff_grey__" + video_name, fourcc, fps, (1280, 720))

    while cap.isOpened():
        contours = find_countours(frame1, frame2)

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        big_contours_counter = 0
        all_contours_size = 0
        big_contours_size = 0

        # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        for contour in contours:
            all_contours_size += cv2.contourArea(contour)
            if cv2.contourArea(contour) < 900:
                pass
            else:
                big_contours_counter += 1
                big_contours_size += cv2.contourArea(contour)
                # cv2.drawContours(frame1, contours, -1, (255, 0, 0), 2)

        output.write(gray)

        big_contours_counts.append(big_contours_counter)
        all_contours_counts.append(len(contours))
        big_contours_sizes.append(big_contours_size)
        all_contours_sizes.append(all_contours_size)
        big_contours_mul.append(big_contours_size * big_contours_counter)
        all_contours_mul.append(all_contours_size * len(contours))

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break

        # print("Frame %s/%s" % (curr_frame, frame_count))
        sys.stdout.write("\r Frame %s/%s" % (curr_frame, frame_count))
        sys.stdout.flush()
        curr_frame += 1

    output.release()
    cap.release()
    print("\n")
    print("Video analysed.\n")

    if not os.path.exists('out/' + video_name):
        os.makedirs('out/' + video_name)

    np.savetxt('./out/' + video_name + '/all_contours_counts.txt', all_contours_counts, fmt='%d')
    np.savetxt('./out/' + video_name + '/big_contours_counts.txt', big_contours_counts, fmt='%d')
    np.savetxt('./out/' + video_name + '/all_contours_sizes.txt', all_contours_sizes, fmt='%d')
    np.savetxt('./out/' + video_name + '/big_contours_sizes.txt', big_contours_sizes, fmt='%d')
    np.savetxt('./out/' + video_name + '/all_contours_mul.txt', all_contours_mul, fmt='%d')
    np.savetxt('./out/' + video_name + '/big_contours_mul.txt', big_contours_mul, fmt='%d')

    return frame_count, fps


def load_importance(video_name):
    return {
        'big_contours_counts': np.loadtxt('./out/'+video_name+'/all_contours_counts.txt', dtype=int),
        'all_contours_counts': np.loadtxt('./out/'+video_name+'/big_contours_counts.txt', dtype=int),
        'big_contours_sizes': np.loadtxt( './out/'+video_name+'/all_contours_sizes.txt', dtype=int),
        'all_contours_sizes': np.loadtxt( './out/'+video_name+'/big_contours_sizes.txt', dtype=int),
        'big_contours_mul': np.loadtxt(   './out/'+video_name+'/all_contours_mul.txt', dtype=int),
        'all_contours_mul': np.loadtxt(   './out/'+video_name+'/big_contours_mul.txt', dtype=int),
    }


def plot_two(imp, key1, key2, c1, c2):
    plt.plot(np.array(imp[key1]), c=c1, label=key1)
    plt.plot(np.array(imp[key2]), c=c2, label=key2)
    plt.legend(loc='upper left')
    plt.show()


def select_important_parts(imp, key, reduction, sma_size):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    # Compute moving average.
    sma = moving_average(np.array(imp[key]), sma_size)

    # Shift and and zero padding
    padding_size = int((FRAME_COUNT - len(sma)) / 2)
    sma = np.concatenate([np.zeros(padding_size), sma, np.zeros(padding_size)])

    # Get threshold.
    sorted_sma = np.sort(sma)
    threshold = sorted_sma[int(len(sorted_sma) * (1 - reduction))]

    # # Remove extremes
    # extreme_threshold = sorted_sma[int(len(sorted_sma) * 0.998)]
    # sma = np.where(sma > extreme_threshold, 0, sma)

    # Eval output
    norm = np.copy(sma)
    norm /= np.max(np.abs(sma), axis=0)
    print(norm.shape)
    np.savetxt(EVAL_OUTPUT_PATH, norm, fmt='%d')
    plt.plot(norm)
    plt.show()


    visual_mask = np.where(sma > threshold, threshold, 0)
    binary_mask = np.where(sma > threshold, True, False)

    return sma, binary_mask, visual_mask


def create_video_summarization(video_name, mask, st):
    cap = cv2.VideoCapture('./in/'+video_name)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    curr_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # store timestamp of current time
    # ts = datetime.datetime.now().timestamp()

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    output = cv2.VideoWriter("./out/" + "sum_" + str(st) + "_" + video_name, fourcc, fps, (1280, 720))

    while cap.isOpened():

        writing = "Skip"
        if curr_frame >= len(mask):
            break
        if mask[curr_frame]:
            if frame is not None:
                # image = cv2.resize(frame, (1280, 720))
                output.write(frame)
                writing = "Write"
            else:
                writing = "None"

        sys.stdout.write("\r Frame %s/%s  %s" % (curr_frame, frame_count, writing))
        sys.stdout.flush()

        curr_frame += 1
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    output.release()


if __name__ == '__main__':
    start = time.time()
    print("start timer for preprocesing")

    # evaluation_output path
    EVAL_OUTPUT_PATH = 'eval_file_0or1.np'
    # input video name - put the video in ./in/yourvideo.mp4
    VIDEO_NAME = 'workout-last.mp4'
    REDUCTION = 0.05
    SMA_SIZE = 60
    IMPORTANCE_METER = 'all_contours_sizes'

    # Select from:
    # 'big_contours_counts'
    # 'all_contours_counts'
    # 'big_contours_sizes'
    # 'all_contours_sizes'
    # 'big_contours_mul'
    # 'all_contours_mul'

    if not os.path.exists('out/' + VIDEO_NAME):
        FRAME_COUNT, FPS = get_importance(VIDEO_NAME)
    else:
        cap = cv2.VideoCapture('./in/' + VIDEO_NAME)
        FPS = cap.get(cv2.CAP_PROP_FPS)
        FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    importance = load_importance(VIDEO_NAME)

    end = time.time()
    print(end - start)

    plot_two(importance, 'all_contours_counts', 'big_contours_counts', 'r', 'b')
    plot_two(importance, 'all_contours_sizes', 'big_contours_sizes', 'g', 'm')
    plot_two(importance, 'all_contours_mul', 'big_contours_mul', 'y', 'c')


    start = time.time()
    print("start timer for cutting")


    # Create highlight mask
    SMA, binary_cut_mask, visual_cut_mask = select_important_parts(importance, IMPORTANCE_METER, REDUCTION, SMA_SIZE)



    importance['SMA_contours_mul'] = SMA
    importance['visual_cut_mask'] = visual_cut_mask

    plot_two(importance, 'all_contours_mul', 'SMA_contours_mul', 'y', 'c')
    plot_two(importance, 'SMA_contours_mul', 'visual_cut_mask', 'c', 'r')

    stamp = '_'+str(REDUCTION)+'_'+str(SMA_SIZE)+'_'+IMPORTANCE_METER+'_'
    create_video_summarization(VIDEO_NAME, binary_cut_mask, stamp)

    end = time.time()
    print(end - start)
