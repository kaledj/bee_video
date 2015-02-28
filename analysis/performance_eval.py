import matplotlib.pyplot as plt
from source.bgsub_mog import bgsub
import os


def main():
    test_videoname = 'video1.mkv'
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

if __name__ == '__main__':
    main()