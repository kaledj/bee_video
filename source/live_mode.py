import re
from ftplib import FTP
import os
from kalman_track import App
import cv2
import time
import tools

files = []
dirs = []

arrivals = []
departures = []

def main():
    user = 'bee'
    pw = 'cs.13,bee'
    ftp = FTP('cs.appstate.edu', user, pw)

    running = True
    lastVideo = None
    lastWait = 256
    app = App()
    while running:
        # get most recent day directory
        ftp.cwd('/usr/local/bee/beemon/pit1')
        ret = ftp.retrlines('LIST', splitDirLine)
        sortDirsByDate(dirs)
        newestDir = dirs[0]
        ftp.cwd("{0}/video".format(newestDir))

        # get most recent video file
        ret = ftp.retrlines('LIST', splitFileLine)    
        sortFilesByTime(files)
        newestFile = files[0]
        if newestFile == lastVideo:
            waitTime = lastWait * 2
            print("Waiting for {0}ms for next video".format(waitTime))
            if tools.handle_keys(waitTime) == 1: 
                return "Test"
            lastWait = waitTime
            continue
        else:
            lastWait = 256

        with open('tempfile.h264', 'wb') as tempfile:
            ret = ftp.retrbinary("RETR %s" % newestFile, tempfile.write)
            print(ret)

        app.openNewVideo('tempfile.h264')
        cv2.namedWindow('Tracking')
        cv2.namedWindow('Mask')
        app.run()
        arrivals.append(app.arrivals)
        departures.append(app.departures)
        print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))

        os.remove('tempfile.h264')
        del(files[:])
        del(dirs[:])
        lastVideo = newestFile
    ftp.quit()

def splitDirLine(string):
    directoryName = string.split()[-1]
    if re.match("^\d{2}-\d{2}-\d{4}$", directoryName):
        dirs.append(directoryName)

def sortDirsByDate(dirs):
    # Sort by day
    dirs.sort(key=lambda dirName: dirName.split('-')[0], reverse=True)
    # Sort by month
    dirs.sort(key=lambda dirName: dirName.split('-')[1], reverse=True)
    # Sort by year
    dirs.sort(key=lambda dirName: dirName.split('-')[2], reverse=True)

def splitFileLine(string):
    fileName = string.split()[-1]
    if re.match("^\d{2}-\d{2}-\d{4}_\d{2}:\d{2}:\d{2}.h264$", fileName):
        files.append(fileName)

def sortFilesByTime(files):
    # Sort by second
    files.sort(key=lambda fileName: fileName.split('_')[1].split(':')[2], reverse=True)
    # Sort by minute
    files.sort(key=lambda fileName: fileName.split('_')[1].split(':')[1], reverse=True)
    # Sort by hour
    files.sort(key=lambda fileName: fileName.split('_')[1].split(':')[0], reverse=True)

if __name__ == '__main__':
    print(main())
    print(arrivals, departures)