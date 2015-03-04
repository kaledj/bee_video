import time
import os
import matplotlib.pyplot as plt
from source.bgsub_mog import bgsub, cascade_detect
from multiprocessing import Process, Lock, Queue
from collections import deque


def eval_ROC(test_videoname):
    detection_rates = {'tp': [], 'fp': []}
    for i in xrange(10):
        print("Testing with threshold {0}".format(2**i))
        tp_rate, fp_rate = bgsub(os.path.basename(test_videoname), 2**i)
        detection_rates['tp'].append(tp_rate)
        detection_rates['fp'].append(fp_rate)
    plt.figure()
    plt.plot(detection_rates['fp'], detection_rates['tp'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print(detection_rates)
    exit()
    

def eval_PR(test_videoname, threshold, results_queue, threadlock, algorithm):
    if algorithm == 'bgsub':
        precision_rate, recall_rate = bgsub(os.path.basename(test_videoname), threshold, quiet=True)
    elif algorithm == 'cascade':
        precision_rate, recall_rate = cascade_detect(os.path.basename(test_videoname), threshold, quiet=True)
    else:
        raise Exception('Invalid algorithm for detection: {0}'.format(algorithm))
    try:
        threadlock.acquire()
        print('P: {0} R: {1}'.format(precision_rate, recall_rate))
        results_queue.put((precision_rate, recall_rate))
    finally:
        threadlock.release()


def main():
    videofile = 'whitebg.h264'
    # eval_ROC('video1.mkv')
    # eval_PR('whitebg.h264')
    time.clock()

    results_q = Queue()
    lock = Lock()
    procs = deque()
    for thresh in xrange(1):
        proc = Process(target=eval_PR, args=(videofile, int(1.4**thresh), results_q, lock, 'bgsub'), name=str(thresh))
        procs.append(proc)
        proc.start()
    for proc_obj, join_func in ((proc, proc.join) for proc in procs):
        join_func()
        print("Process '{0}' joined.".format(proc_obj.name))
    print("{0} seconds elapsed".format(time.clock()))

    results = []
    recall_rates = []
    precision_rates = []
    while not results_q.empty():
        results.append(results_q.get())
    results.sort(key=lambda x: x[1])
    for prec, recall in ((item[0], item[1]) for item in results):
        precision_rates.append(prec)
        recall_rates.append(recall)
    plt.figure()
    plt.title('Precision-Recall Curve for Bee Detection')
    plt.plot(recall_rates, precision_rates, label='Background sub. algorithm')

    #######################################
    results_q = Queue()
    lock = Lock()
    procs = deque()
    for thresh in xrange(30):
        proc = Process(target=eval_PR, args=(videofile, int(thresh), results_q, lock, 'cascade'), name=str(thresh))
        procs.append(proc)
        proc.start()
    for proc_obj, join_func in ((proc, proc.join) for proc in procs):
        join_func()
        print("Process '{0}' joined.".format(proc_obj.name))
    print("{0} seconds elapsed".format(time.clock()))

    results = []
    recall_rates = []
    precision_rates = []
    while not results_q.empty():
        results.append(results_q.get())
    results.sort(key=lambda x: x[1])
    for prec, recall in ((item[0], item[1]) for item in results):
        precision_rates.append(prec)
        recall_rates.append(recall)
    plt.plot(recall_rates, precision_rates, label='Cascade algorithm')
    ########################################################################

    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()