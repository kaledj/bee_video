import sys
# PREPEND the source to the path so that this package can access it
sys.path.insert(0, 'C:/Users/kaledj/Projects/bee_video_dev')

# System imports
import cv2
import numpy as np

# Project imports
import source.lk_track

def area_precision_recall(detector_response, actual):
    if type(actual) is str:
        actual_img = cv2.imread(actual, flags=-1)
    else:
        actual_img = actual
    assert actual_img.shape == detector_response.shape, \
        "Actual shape: {0} Got: {1}".format(actual_img.shape, detector_response.shape)
    assert all_binary(detector_response), np.array_repr(np.unique(detector_response))
    assert all_binary(actual_img), actual

    pr = area_precision(detector_response, actual_img)
    rec = area_recall(detector_response, actual_img)
    return pr, rec

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


def area_recall(detector_response, actual_img, weightedSoFar=0, totalSoFar=0, eps=1e-7):
    tp = true_positive(detector_response, actual_img)
    p = np.sum(actual_img)
    rec = np.float64(tp) / (p + eps)
    weighted = rec * p
    weightedSoFar += weighted
    totalSoFar += p
    return np.float64(weightedSoFar) / (totalSoFar + eps)


def area_precision(detector_response, actual_img, weightedSoFar=0, totalSoFar=0, eps=1e-7):
    p_d = np.sum(detector_response)
    tp = true_positive(detector_response, actual_img)
    fp = false_postive(detector_response, actual_img)
    prec = np.float64(tp) / (tp + fp + eps)
    weighted = prec * p_d
    weightedSoFar += weighted
    totalSoFar += p_d
    return np.float64(weightedSoFar) / (totalSoFar + eps)


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
    print area_precision(test1, test2)