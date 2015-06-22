from kalman_track import App
import cv2
from time import clock
from matplotlib import pyplot

def main():
    print("OpenCV version: {0}".format(cv2.__version__))
    clock()
    videos = []
    # videos.append("../videos/crowded_4pm.h264")
    # videos.append("../videos/crowded_7am.h264")
    # videos.append("../videos/rpi2.h264")
    # videos.append("../videos/video1.mkv")
    # videos.append("../videos/whitebg.h264")
    videos.append("../videos/newhive_noshadow3pm.h264")
    # videos.append("../videos/newhive_shadow2pm.h264")


    for video_src in videos:
        # Calculate area histograms
        h, w = 2, 2    
        f, axarr = pyplot.subplots(h, w)
        for i in xrange(h):
            for j in xrange(w):
                app = App(video_src, invisible=True, bgsub_thresh=2**(i*w+j+2))
                app.run()
                areas = app.areas
                axarr[i, j].set_title(
                    "Threshold: {0}  Detections: {1}".format(app.threshold, len(areas)))
                axarr[i, j].hist(areas, 50, range=(0, 2000))
                axarr[i, j].set_xlabel("Area, pixels")
                axarr[i, j].set_ylabel("Occurances")
                cv2.destroyAllWindows()
                print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))
                print("{0} seconds elapsed.".format(clock()))
        pyplot.suptitle('Areas and Counts of Detections in {0}'.format(video_src))
        
        f, axarr = pyplot.subplots(h, w)
        for i in xrange(h):
            for j in xrange(w):
                app = App(video_src, invisible=True, bgsub_thresh=2**(i*w+j+6))
                app.run()
                areas = app.areas
                axarr[i, j].set_title(
                    "Threshold: {0}  Detections: {1}".format(app.threshold, len(areas)))
                axarr[i, j].hist(areas, 50, range=(0, 2000))
                axarr[i, j].set_xlabel("Area, pixels")
                axarr[i, j].set_ylabel("Occurances")
                cv2.destroyAllWindows()
                print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))
                print("{0} seconds elapsed.".format(clock()))
        pyplot.suptitle('Areas and Counts of Detections in {0}'.format(video_src))

        pyplot.show()

if __name__ == '__main__':
    main()
