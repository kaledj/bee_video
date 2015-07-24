import re
from ftplib import FTP
import os
from kalman_track import App
import cv2
import time
from threading import Timer
import tools
import sys

files = []
dirs = []

totalFlow = 0
arrivals = []
departures = []
times = []
totalFlowSamples = []

def main():
    global totalFlow
    log = open('Log.txt', 'w')
    lastTimeStamp = -1
    time.clock()
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
                break
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
        totalFlow += app.arrivals
        totalFlow -= app.departures

        # Log
        timeStr = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        logStr = "{0} {1}\n".format(timeStr, str(totalFlow))
        log.write(logStr)
        
        arrivals.append(app.arrivals)
        departures.append(app.departures)

        # print("Arrivals: {0} Departures: {1}".format(app.arrivals, app.departures))

        os.remove('tempfile.h264')
        del(files[:])
        del(dirs[:])
        lastVideo = newestFile
    ftp.quit()
    log.close()

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

def logCurrentFlow(log):
    log.write("Logging {0} total flow".format(totalFlow))
    totalFlowSamples.append(totalFlow)
    Timer(10, logCurrentFlow, log).start()

if __name__ == '__main__':
    main()
    print(arrivals, departures)
    sys.exit()
