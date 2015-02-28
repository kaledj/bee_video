
import cv2
import numpy as np


def compare_response_to_truth(detector_response, actual):
    if type(actual) is str:
        actual_img = cv2.imread(actual, flags=-1)
    else:
        actual_img = actual
    assert actual_img.shape == detector_response.shape
    assert all_binary(detector_response), np.array_repr(np.unique(detector_response))
    assert all_binary(actual_img), actual

    pos_response_idx = (detector_response == 1)
    tp = np.sum(actual_img[pos_response_idx])
    fp = np.sum(actual_img[pos_response_idx] == 0)
    return tp, fp


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