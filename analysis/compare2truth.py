import sys
# PREPEND the source to the path so that this package can access it
sys.path.insert(0, 'C:/Users/kaledj/Projects/bee_video_dev')

# System imports
import cv2
import numpy as np

# Project imports
import source.lk_track

def compare_response_to_truth(detector_response, actual, cascade=False, thresh=10):
    if type(actual) is str:
        actual_img = cv2.imread(actual, flags=-1)
    else:
        actual_img = actual
    assert actual_img.shape == detector_response.shape, \
        "Actual shape: {0} Got: {1}".format(actual_img.shape, detector_response.shape)
    assert all_binary(detector_response), np.array_repr(np.unique(detector_response))
    assert all_binary(actual_img), actual

    tp = true_positive(detector_response, actual_img, cascade, thresh)
    fp = false_postive(detector_response, actual_img, cascade, thresh)
    fn = false_negative(detector_response, actual_img, cascade, thresh)
    return tp, fp, fn


def true_positive(detector_response, actual_img, cascade=False, thresh=10):
    pos_response_idx = (detector_response == 1)
    return np.sum(actual_img[pos_response_idx])


def false_postive(detector_response, actual_img, cascade=False, thresh=10):
    pos_response_idx = (detector_response == 1)
    if cascade:
        return np.sum(actual_img[pos_response_idx] == 0) / 4
    else:
        return np.sum(actual_img[pos_response_idx] == 0)


def false_negative(detector_response, actual_img, cascade=False, thresh=10):
    neg_response_idx = (detector_response == 0)
    return np.sum(actual_img[neg_response_idx])


def all_binary(array):
    return np.all(((array == 1).astype(np.uint8) + (array == 0).astype(np.uint8)) == 1)


if __name__ == '__main__':
    # Should have 3 TP, 2 FP
    test1 = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [1, 1, 1]])

    test2 = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1]])
    print compare_response_to_truth(test1, test2)