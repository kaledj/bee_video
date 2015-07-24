import sys
# PREPEND the source to the path so that this package can access it
sys.path.insert(0, 'C:/Users/kaledj/Projects/bee_video_dev')

import time
import os
from collections import deque
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock, Queue

from source.bgsub_mog import bgsub, cascade_detect

NUM_ITERS = 32
QUIET = True

# def eval_ROC(test_videoname):
#     detection_rates = {'tp': [], 'fp': []}
#     for i in xrange(10):
#         print("Testing with threshold {0}".format(2**i))
#         tp_rate, fp_rate = bgsub(os.path.basename(test_videoname), 2**i)
#         detection_rates['tp'].append(tp_rate)
#         detection_rates['fp'].append(fp_rate)
#     plt.figure()
#     plt.plot(detection_rates['fp'], detection_rates['tp'])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.show()
#     print(detection_rates)
#     exit()
    

def eval_PR(test_videoname, threshold, results_queue, threadlock, algorithm):
    if algorithm == 'bgsub':
        precision_rate, recall_rate = bgsub(os.path.basename(test_videoname), 
            threshold, quiet=QUIET)
    elif algorithm == 'cascade':
        precision_rate, recall_rate = cascade_detect(os.path.basename(test_videoname), 
            threshold, quiet=QUIET)
    else:
        raise Exception('Invalid algorithm for detection: {0}'.format(algorithm))
    try:
        threadlock.acquire()
        print('P: {0} R: {1}'.format(precision_rate, recall_rate))
        results_queue.put((precision_rate, recall_rate))
    finally:
        threadlock.release()


def run_multithreaded(videofile, algorithm, threshold_step):
    time.clock()
    results_q = Queue()
    lock = Lock()
    procs = deque()
    for thresh in xrange(NUM_ITERS):
        if algorithm is 'bgsub':
            threshold = threshold_step ** thresh
        else:
            threshold = thresh
        proc = Process(target=eval_PR, args=(videofile, threshold, results_q, 
            lock, algorithm), name=str(thresh))
        procs.append(proc)
        print("Starting process {0} with {1} algorithm and threshold={2}".format(
            proc.name, algorithm, threshold))
        proc.start()
    for proc_obj, join_func in ((proc, proc.join) for proc in procs):
        join_func()
        print("Process '{0}' using algorithm '{1}' joined.".format(proc_obj.name, algorithm))
    print("{0} seconds elapsed".format(time.clock()))
    # Organize results
    results = []
    recall_rates = []
    precision_rates = []
    while not results_q.empty():
        results.append(results_q.get())
    results.sort(key=lambda x: x[1])
    for prec, recall in ((item[0], item[1]) for item in results):
        precision_rates.append(prec)
        recall_rates.append(recall)
    return precision_rates, recall_rates


def main():
    videofile = 'whitebg.h264'
    # videofile = 'newhive_noshadow3pm.h264'

    # Set up and create the figure
    plt.figure()
    plt.title('Precision-Recall Curve for Bee Detection')

    # Run with bgsub algorithm 
    # recall_rates, precision_rates = run_multithreaded(videofile, 'bgsub', 
        # threshold_step=1.3)    
    # plt.plot(recall_rates, precision_rates)

    # Run with cascade detection algorithm
    recall_rates, precision_rates = run_multithreaded(videofile, 'cascade', 
        threshold_step=1)  
    plt.plot(precision_rates, recall_rates)

    # Add labels and a diagonal dashed line
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show() 


if __name__ == '__main__':
    main()