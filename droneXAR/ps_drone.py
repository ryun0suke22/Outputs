"""
2018-3-14, 15, 16  Aizu Tokai Hackathon

ARDrone read ARMaker and print POKEMON

"""


"""
# Dependencies: a POSIX OS, openCV2 for video-support.

# Base-program of the PS-Drone API: "An open and enhanced API for universal control of the Parrot AR.Drone 2.0 quadcopter."

Fork of www.playsheep.de/drone and https://sourceforge.net/projects/ps-drone/

# (w)+(c) J. Philipp de Graaff, www.playsheep.de, drone@playsheep.de, 2012-2014


##########
# LICENCE:
#   Artistic License 2.0 as seen on http://opensource.org/licenses/artistic-license-2.0 (retrieved December 2014)
#   If the terms of this license do not permit the full use that you propose to make of PS-Drone, please contact me for a
#   different licensing arrangement.
#   Visit www.playsheep.de/drone or see the PS-Drone-API-documentation for an abstract from the Artistic License 2.0.

"""
from __future__ import print_function

import threading
import select
import socket
import time
import tempfile
import multiprocessing
import struct
import os
import sys
import thread
import signal
import subprocess
import numpy as np
from PIL import Image


import pygame
from pygame.locals import *

if os.name == 'posix':
    import termios
    import fcntl  # for getKey(), ToDo: Reprogram for Windows

commitsuicideV, showVid, vCruns, lockV, debugV = False, False, False, threading.Lock(), False  # Global variables for video-decoding
# Global variables for NavDava-decoding
offsetND, suicideND, commitsuicideND = 0, False, False


class Drone(object):
    """
    Start and stop using the drone ###=-
    Bootup and base configuration
    """
   # vidPipePath = ""
    
    vidTemp = 1
    def __init__(self):
        self._Version = "2.0.1"
        self._lock = threading.Lock()  # To prevent semaphores
        self._startTime = time.time()
        self._speed = 0.2     # Default drone moving speed in percent.
        # Shows all sent commands (but not the keepalives)
        self.showCommands = False
        self.debug = False    # Shows some additional debug information
        self.valueCorrection = False
        # use this value, if not checked by getSelfRotation()
        self.selfRotation = 0.0185
        # when there is a communication-problem, drone will land or not
        self.stopOnComLoss = False
        #self.vidTemp = 0
        # Drone communication variables
        self.DroneIP = "192.168.1.1"
        self.NavDataPort = 5554
        self.VideoPort = 5555
        self.CmdPort = 5556
        self.CTLPort = 5559

        # NavData variables
        self._NavData = ""
        self._State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._NavDataCount = 0
        self._NavDataTimeStamp = 0.0
        self._NavDataDecodingTime = 0.0
        self._NoNavData = False

        # Video variables
        self._VideoImage = None
        self._VideoImageCount = 0
        self._VideoDecodeTimeStamp = 0
        self._VideoDecodeTime = 0
        self._VideoReady = False
        self._vKey = ""
        self._SaveVideo = False

        # Config variables
        self._ConfigData = []
        self._ConfigDataCount = 0
        self._ConfigDataTimeStamp = 0
        self._ConfigSending = True
        self._ConfigSessionID = "03016321"
        self._ConfigUserID = "0a100407"
        self._ConfigApplicationID = "03016321"
        self.sendConfigSaveMode = False

        # Internal variables
        self._NavDataProcess = ""
        self._VideoProcess = ""
        self._vDecodeProcess = ""
        self._ConfigQueue = []
        self._networksuicide = False
        self._receiveDataRunning = False
        self._sendConfigRunning = False
        self._shutdown = False
        self._pDefaultStr = "\033[0m"
        self._pRedStr = "\033[91m"
        self._pGreenStr = "\033[92m"
        self._pYellowStr = "\033[93m"
        self._pBlueStr = "\033[94m"
        self._pPurpleStr = "\033[95m"
        self._pLineUpStr = "\033[1A"

    # Connect to the drone and start all procedures
    def startup(self):
        # Check for drone in the network and wake it up
        try:
            socket.socket().connect((self.DroneIP, 21))
            socket.socket().close()
        except:
            self.printRed()
            print("Drone is not online")
            self.printDefault()
            sys.exit(9)

        # Internal variables
        # as there are two raw commands, send next steps
        self._CmdCounter = 3
        self._calltime = 0           # to get some time-values to debug

        # send the first four initial-commands to the drone
        # Open network connection
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(0)										# Network should not block
        self._sendrawmsg("\r")											# Wakes up command port
        time.sleep(0.01)
        # Initialising drone as sniffed from datastream demo-tool to AR.Drone
        self._sendrawmsg("AT*PMODE=1,2\rAT*MISC=2,2,20,2000,3000\r")

        # Initialising timed thread(s) for drone communication
        # Opening NavData- and Video- Processes
        self._VidPipePath = tempfile.gettempdir() + "/dronevid-" + str(threading.enumerate()
                                                                        [0])[-12:-2] + "-" + str(time.time())[-7:].replace(".", "") + ".h264"
        self._net_pipes = []
        self._NavData_pipe, navdataChild_pipe = multiprocessing.Pipe()
        self._Video_pipe,   videoChild_pipe = multiprocessing.Pipe()
        self._vdecode_pipe, self._vdecodeChild_pipe = multiprocessing.Pipe()

        self._NavDataProcess = multiprocessing.Process(target=mainloopND, args=(
            self.DroneIP, self.NavDataPort, navdataChild_pipe, os.getpid()))
        self._NavDataProcess.start()
        self._VideoProcess = multiprocessing.Process(target=mainloopV, args=(
            self.DroneIP, self.VideoPort, self._VidPipePath, videoChild_pipe, os.getpid()))
        self._VideoProcess.start()
        self._vDecodeProcess = multiprocessing.Process(
            target=vDecode, args=(self._VidPipePath, self._vdecodeChild_pipe, os.getpid()))
        # There is a third process called "self._vDecodeProcess" for decoding
        # video, initiated and started around line 880

        # Final settings
        # This entry is necessary for the drone's firmware, otherwise the
        # NavData contains just header and footer
        self.useDemoMode(True)
        self.setConfig("custom:session_id", "-all")
        self.getNDpackage(["demo"])

        time.sleep(1)
        # setup Network-thread
        # sometimes they would not start why ever, so TK has to double-check
        while not self._receiveDataRunning or not self._sendConfigRunning or len(self._ConfigQueue):
            if not self._receiveDataRunning:
                self._threadReceiveData = threading.Thread(
                    target=self._receiveData)
                self._threadReceiveData.start()
                time.sleep(0.05)
            if not self._sendConfigRunning:
                self._threadSendConfig = threading.Thread(
                    target=self._sendConfig)
                self._threadSendConfig.start()
                time.sleep(0.05)
            time.sleep(0.01)

    # Clean Shutdown
    def shutdown(self):
        if self._shutdown:
            sys.exit()
        self._shutdown = True
        if self.debug:
            print("Shutdown...")
        self.land()
        self.thrust(0, 0, 0, 0)
        try:
            self._NavData_pipe.send("die!")
        except:
            pass
        self._Video_pipe.send("uninit")
        t = time.time()
        while self._VideoReady and (time.time() - t) < 5:
            time.sleep(0.1)
        try:
            self._Video_pipe.send("die!")
        except:
            pass

        time.sleep(0.5)
        try:
            self._VideoProcess.terminate()
        except:
            pass
        try:
            self._vDecodeProcess.terminate()
        except:
            pass
        try:
            self._NavDataProcess.terminate()
        except:
            pass

        self._stopnetwork()
        try:
            self._threadSendConfig.join()
        except:
            pass
        try:
            self._threadReceiveData.join()
        except:
            pass
        self._keepalive.cancel()
        sys.exit()

# =-
# Make internal variables to external read-only variables ###=-
# =-
    @property
    def Version(self):
        return self._Version

    @property
    def startTime(self):
        return self._startTime

    @property
    def speed(self):
        return self._speed

    @property
    def NavData(self):
        return self._NavData

    @property
    def State(self):
        return self._State

    @property
    def NavDataCount(self):
        return self._NavDataCount

    @property
    def NavDataTimeStamp(self):
        return self._NavDataTimeStamp

    @property
    def NavDataDecodingTime(self):
        return self._NavDataDecodingTime

    @property
    def NoNavData(self):
        return self._NoNavData

    @property
    def VideoImage(self):
        return self._VideoImage

    @property
    def VideoImageCount(self):
        return self._VideoImageCount

    @property
    def VideoDecodeTimeStamp(self):
        return self._VideoDecodeTimeStamp

    @property
    def VideoDecodeTime(self):
        return self._VideoDecodeTime

    @property
    def VideoReady(self):
        return self._VideoReady

    @property
    def SaveVideo(self):
        return self._SaveVideo

    @property
    def ConfigData(self):
        return self._ConfigData

    @property
    def ConfigDataCount(self):
        return self._ConfigDataCount

    @property
    def ConfigDataTimeStamp(self):
        return self._ConfigDataTimeStamp

    @property
    def ConfigSending(self):
        return self._ConfigSending

    @property
    def ConfigSessionID(self):
        return self._ConfigSessionID

    @property
    def ConfigUserID(self):
        return self._ConfigUserID

    @property
    def ConfigApplicationID(self):
        return self._ConfigApplicationID

# =-
# Drone commands ###=-
# =-
    # Commands for configuration
    # change some value
    # e.g. drone.setConfig(control:altitude_max","5000")
    def setConfig(self, name, value):
        # Note: changes are not immediately and could take some time
        self._ConfigQueue.append([str(name), str(value), False])

    # change some value and send the configuration Identifier (sendConfigIDs)
    # ahead
    def setMConfig(self, name, value):								# Usage like setConfig
        # Note: changes are not immediately and could take some time
        self._ConfigQueue.append([str(name), str(value), True])

    # get actual configuration
    def getConfig(self):											# Stored in "ConfigData"
        # Wow, that is new, was not necessary before
        self.at("CTRL", [5, 0])
        # Note: Actual configuration data will be received after setting...
        self.at("CTRL", [4, 0])
        if self.showCommands:
            # ... automatically. An update will take up to 0.015 sec)
            self._calltime = time.time()

    # setting IDs to store Konfigurations for later
    def setConfigSessionID(self, *args):
        try:
            value = float(args[0])
            self._ConfigSessionID = normalLen8(value)
            self.setConfig("custom:session_id", self._ConfigSessionID)
        except:
            return (self._ConfigSessionID)

    def setConfigUserID(self, *args):
        try:
            value = float(args[0])
            self._ConfigUserID = normalLen8(value)
            self.setConfig("custom:profile_id", self._ConfigUserID)
        except:
            return (self._ConfigUserID)

    def setConfigApplicationID(self, *args):
        try:
            value = float(args[0])
            self._ConfigApplicationID = normalLen8(value)
            self.setConfig("custom:application_id", self._ConfigApplicationID)
        except:
            return (self._ConfigApplicationID)

    def setConfigAllID(self):
        self.setConfig("custom:session_id", self._ConfigSessionID)
        self.setConfig("custom:profile_id", self._ConfigUserID)
        self.setConfig("custom:application_id", self._ConfigApplicationID)

    # Reminds the drone which IDs it has to use (important for e.g. switch
    # cameras)
    def sendConfigIDs(self):
        self.at("CONFIG_IDS", [
                self._ConfigSessionID, self._ConfigUserID, self._ConfigApplicationID])

    # Calibration
    def trim(self):
        self.at("FTRIM", [])

    def mtrim(self):
        self.at("CALIB", [0])

    def mantrim(self, thetaAngle, phiAngle, yawAngle):		# manual Trim
        if self.valueCorrection:
            try:
                thetaAngle = float(thetaAngle)
            except:
                thetaAngle = 0.0
            try:
                phiAngle = float(phiAngle)
            except:
                phiAngle = 0.0
            try:
                yawAngle = float(yawAngle)
            except:
                yawAngle = 0.0
        self.at("MTRIM", [thetaAngle, phiAngle, yawAngle])  # floats

    def getSelfRotation(self, wait):
        if self.valueCorrection:
            try:
                wait = float(wait)
            except:
                wait = 1.0
        reftime = time.time()
        # detects the self-rotation-speed of the yaw-sensor
        oangle = self._NavData["demo"][2][2]
        time.sleep(wait)
        self.selfRotation = (
            self._NavData["demo"][2][2] - oangle) / (time.time() - reftime)
        return self.selfRotation

    # Movement
    # Default speed of movement
    def setSpeed(self, *speed):
        try:
            self._speed = self._checkSpeedValue(*speed)
        except:
            pass
        return self._speed

    # Absolute movement in x, y and z-direction and rotation
    # Absolute movement in x, y and z-direction and rotation
    def move(self, leftright, backwardforward, downup, turnleftright):
        if self.valueCorrection:
            try:
                leftright = float(leftright)
            except:
                leftright = 0.0
            try:
                backwardforward = float(backwardforward)
            except:
                backwardforward = 0.0
            try:
                downup = float(downup)
            except:
                downup = 0.0
            try:
                turnleftright = float(turnleftright)
            except:
                turnleftright = 0.0
        if leftright > 1.0:
            leftright = 1.0
        if leftright < -1.0:
            leftright = -1.0
        if backwardforward > 1.0:
            backwardforward = 1.0
        if backwardforward < -1.0:
            backwardforward = -1.0
        if downup > 1.0:
            downup = 1.0
        if downup < -1.0:
            downup = -1.0
        if turnleftright > 1.0:
            turnleftright = 1.0
        if turnleftright < -1.0:
            turnleftright = -1.0
        self.at(
            "PCMD", [3, leftright, -backwardforward, downup, turnleftright])

    # Relative movement to controller in x, y and z-direction and rotation
    def relMove(self, leftright, backwardforward, downup, turnleftright, eastwest, northturnawayaccuracy):
        if self.valueCorrection:
            try:
                leftright = float(leftright)
            except:
                leftright = 0.0
            try:
                backwardforward = float(backwardforward)
            except:
                backwardforward = 0.0
            try:
                downup = float(downup)
            except:
                downup = 0.0
            try:
                turnleftright = float(turnleftright)
            except:
                turnleftright = 0.0
        if leftright > 1.0:
            leftright = 1.0
        if leftright < -1.0:
            leftright = -1.0
        if backwardforward > 1.0:
            backwardforward = 1.0
        if backwardforward < -1.0:
            backwardforward = -1.0
        if downup > 1.0:
            downup = 1.0
        if downup < -1.0:
            downup = -1.0
        if turnleftright > 1.0:
            turnleftright = 1.0
        if turnleftright < -1.0:
            turnleftright = -1.0
        self.at("PCMD_MAG", [1, leftright, -backwardforward,
                             downup, turnleftright, eastwest, northturnawayaccuracy])

    # Stop moving
    def hover(self):
        self.at("PCMD", [0, 0.0, 0.0, 0.0, 0.0])

    def stop(self):  # Hammertime !
        self.hover()

    # Basic movements
    def moveLeft(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(-self._checkSpeedValue(speed), 0.0, 0.0, 0.0)

    def moveRight(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(self._checkSpeedValue(speed), 0.0, 0.0, 0.0)

    def moveForward(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, self._checkSpeedValue(speed), 0.0, 0.0)

    def moveBackward(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, -self._checkSpeedValue(speed), 0.0, 0.0)

    def moveUp(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, 0.0, self._checkSpeedValue(speed), 0.0)

    def moveDown(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, 0.0, -self._checkSpeedValue(speed), 0.0)

    def turnLeft(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, 0.0, 0.0, -self._checkSpeedValue(speed))

    def turnRight(self, *args):
        try:
            speed = args[0]
        except:
            speed = self._speed
        self.move(0.0, 0.0, 0.0, self._checkSpeedValue(speed))

    # Lets the drone rotate defined angle
    # BUG: does not work with 180deg. turns
    # ToDo: Should be able to stop in case of failures
    def turnAngle(self, ndir, speed, *args):
        # get the source/current (original) angle
        opos = self._NavData["demo"][2][2]
        npos = opos + ndir        # calculate the destination (new) angle
        # to make sure, that the jump from -180 to 180 will...
        minaxis = opos
        maxaxis = opos         # ...be correctly handled
        speed = self._checkSpeedValue(speed)
        ospeed = speed         # stores the given speed-value
        reftime = time.time()
        accurateness = 0
        try:
            accurateness = args[0]
        except:
            pass
        if accurateness <= 0:
            # Destination angle can differ +/- this value (not demo-mode)
            accurateness = 0.005
            if self._State[10]:
                # Destination angle can differ +/- this value in demo-mode
                accurateness = 0.1
        stop = False
        while not stop:
            ndc = self._NavDataCount						# wait for the next NavData-package
            while ndc == self._NavDataCount:
                time.sleep(0.001)
            # trys to recalibrate, causing moving sensor-values around 0.0185
            # deg/sec
            kalib = (time.time() - reftime) * self.selfRotation
            cpos = self._NavData["demo"][2][2]				# get the current angle
            if minaxis > cpos:
                minaxis = cpos		# set the minimal seen angle
            if maxaxis < cpos:
                maxaxis = cpos		# set the maximal seen angle
            if cpos - minaxis >= 180:
                cpos = cpos - 360		# correct the angle-value if necessary...
            elif maxaxis - cpos >= 180:
                cpos = cpos + 360		# ...for an easier calculation
            # the closer to the destination the slower the drone turns
            speed = abs(cpos - npos + kalib) / 10.0
            if speed > ospeed:
                speed = ospeed		# do not turn faster than recommended
            if speed < 0.05:
                # too slow turns causes complications with calibration
                speed = 0.05
            self._speed = speed
            if cpos > (npos + kalib):
                self.turnLeft()		# turn left, if destination angle is lower
            else:
                self.turnRight()  # turn right if destination angle is higher
            # if angle is reached...
            if cpos < (npos + kalib + accurateness) and cpos > (npos + kalib - accurateness):
                self.stop()									# ...stop turning
                time.sleep(0.01)
                stop = True
        return(True)

    def takeoff(self):
        self.at("REF", [290718208])  # 290718208=10001010101000000001000000000

    def land(self):
        self.at("REF", [290717696])  # 290717696=10001010101000000000000000000

    # NavData commands
    # Switches to Demo- or Full-NavData-mode
    def useDemoMode(self, value):
        if value:
            self.setConfig("general:navdata_demo", "TRUE")
        else:
            self.setConfig("general:navdata_demo", "FALSE")

    def useMDemoMode(self, value):
        if value:
            self.setMConfig("general:navdata_demo", "TRUE")
        else:
            self.setMConfig("general:navdata_demo", "FALSE")

    def getNDpackage(self, packets):
        self._NavData_pipe.send(("send", packets))

    def addNDpackage(self, packets):
        self._NavData_pipe.send(("add", packets))

    def delNDpackage(self, packets):
        self._NavData_pipe.send(("block", packets))

    def reconnectNavData(self):
        self._NavData_pipe.send("reconnect")

    # Video & Marker commands
    # This makes the drone fly around and follow 2D tags which the camera is
    # able to detect.
    def aflight(self, flag):
        self.at("AFLIGHT", [flag])  # Integer: 1: start flight, 0: stop flight

    def slowVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("slowVideo")
        else:
            self._Video_pipe.send("fastVideo")

    def midVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("midVideo")
        else:
            self._Video_pipe.send("fastVideo")

    def fastVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("fastVideo")
        else:
            self._Video_pipe.send("slowVideo")

    def saveVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("saveVideo")
        else:
            self._Video_pipe.send("unsaveVideo")

    def startVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("init")
        else:
            self.stopVideo()

    def stopVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("uninit")
        else:
            self.startVideo()

    def showVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("init")
            self._Video_pipe.send("show")
        else:
            self.hideVideo()

    def hideVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self._Video_pipe.send("init")
            self._Video_pipe.send("hide")
        else:
            self.showVideo()

    # Selects which video stream to send on the video UDP port.
    def hdVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self.setMConfig("video:video_codec", "131")
        else:
            self.setMConfig("video:video_codec", "129")

    def sdVideo(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self.setMConfig("video:video_codec", "129")
        else:
            self.setMConfig("video:video_codec", "131")

    def mp4Video(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self.setMConfig("video:video_codec", "128")
        else:
            self.setMConfig("video:video_codec", "129")

    # Selects which video-framerate (in frames per second) to send on the
    # video UDP port.
    def videoFPS(self, fps):
        try:
            int(fps)
            if fps > 60:
                fps = 60
            elif fps < 1:
                fps = 1
            self.setMConfig("video:codec_fps", fps)
        except:
            pass

    # Selects which video-bitrate (in kilobit per second) to send on the video
    # UDP port.
    def videoBitrate(self, bitrate):
        try:
            int(bitrate)
            if bitrate > 20000:
                bitrate = 20000
            if bitrate < 250:
                bitrate = 250
            self.setMConfig("video:bitrate", bitrate)
        except:
            pass

    # Selects which video stream to send on the video UDP port.
    def frontCam(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self.setMConfig("video:video_channel", "0")
        else:
            self.setMConfig("video:video_channel", "1")

    def groundCam(self, *args):
        try:
            do = args[0]
        except:
            do = True
        if do:
            self.setMConfig("video:video_channel", "1")
        else:
            self.setMConfig("video:video_channel", "0")

# Misc commands
    def reset(self):
        if self.NavDataCount > 0 and self.State[31] == 1:
            print("RESET!")
            # 290717952=10001010101000000000100000000
            self.at("REF", [290717952])

    # Controls engines directly, overriding control loops.
    def thrust(self, fl, fr, rl, rr):
        fl *= 2
        if fl > 64000:
            fl = 64000
        elif fl < 0:
            fl = 0
        fr *= 2
        if fr > 64000:
            fr = 64000
        elif fr < 0:
            fr = 0
        rl *= 2
        if rl > 64000:
            rl = 64000
        elif rl < 0:
            rl = 0
        rr *= 2
        if rr > 64000:
            rr = 64000
        elif rr < 0:
            rr = 0

        self.at("PWM", [int(fl), int(fr), int(rr), int(rl)])
        # Seems that integer-values could be between 0 (stop) to 511 (full); more than 511 seem to have no effect.
        # Beware: if using too high values (e.g. floats (>64k ?)), there will be side-effects like restarting other motors, etc.
        # Drone will shut down, if its flight-angle is more than set.

#   Control the drone's LED.
    def led(self, animation, frequency, duration):
        if animation < 21 and frequency > 0 and duration >= 0:
            self.at("LED", [animation, float(frequency), duration])

#   Makes the drone execute a predefined movement (animation).
    def anim(self, animation, duration):
        if animation < 20 and duration >= 0:
            self.at("ANIM", [animation, duration])


# =-
# Low-level Commands ###=-
# =-

    # Upgrading the basic drone commands to low-level drone commands:vid
    # Adding command-number, checking the values, convert 32-bit float to
    # 32-bit integer and put it in quotes
    def at(self, command, params):
        self._lock.acquire()
        paramLn = ""
        if params:
            for p in params:
                if type(p) == int:
                    paramLn += "," + str(p)
                elif type(p) == float:
                    paramLn += "," + \
                        str(struct.unpack("i", struct.pack("f", p))[0])
                elif type(p) == str:
                    paramLn += ",\"" + p + "\""
        msg = "AT*" + command + "=" + str(self._CmdCounter) + paramLn + "\r"
        self._CmdCounter += 1
        self._sendrawmsg(msg)
        self._lock.release()

    # Sending the low-level drone-readable commands to the drone...better do
    # not use
    def _sendrawmsg(self, msg):
        try:
            self._keepalive.cancel()
        except:
            pass
        if self.showCommands:
            if msg.count("COMWDG") < 1:
                print(msg)
        self._sock.sendto(msg, (self.DroneIP, self.CmdPort))
        self._keepalive = threading.Timer(0.1, self._heartbeat)
        self._keepalive.start()


# =-
# Convenient Commands  ###=-
# =-
#    Just add water
    # Checks the battery-status
    def getBattery(self):
        batStatus = "OK"
        batValue = 0
        if self._State[15] == 1:
            batStatus = "empty"
        try:
            batValue = self._NavData['demo'][1]
        except:
            batValue = -1
        return (batValue, batStatus)  # Percent & status ("OK", "empty")

    # Calculates the minor difference between two angles as the drone gives values from -180 to 180...
    # ...so e.g. 170 and -160 are +30 difference and drone will turn to the correct direction
    def angleDiff(self, base, value):
        adiff = ((base + 180) - (value + 180)) % 360
        if adiff > 180:
            adiff -= 360
        return adiff

    # Grabs the pressed key (not yet for Windows)
    # ToDo: Reprogram for Windows
    def getKey(self):
        key = ""
        fd = sys.stdin.fileno()
        if os.name == 'posix':
            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
            try:
                try:
                    key = sys.stdin.read(1)
                except IOError:
                    pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        if os.name == 'nt':
            if msvcrt.kbhit():
                key = msvcrt.getch()
        key += self._vKey
        self._vKey = ""
        return key

    # Drone hops like an excited dog
    def doggyHop(self):
        ospeed = self._speed
        self._speed = 1
        for i in range(0, 4, 1):
            self.moveUp()
            time.sleep(0.20)
            self.moveDown()
            time.sleep(0.20)
        self.hover()
        self._speed = ospeed

    # Drone wags like a happy dog
    def doggyWag(self):
        ospeed = self._speed
        self._speed = 1
        for i in range(0, 4, 1):
            self.moveLeft()
            time.sleep(0.25)
            self.moveRight()
            time.sleep(0.25)
        self.hover()
        self._speed = ospeed

    # Drone nods
    def doggyNod(self):
        ospeed = self._speed
        self._speed = 1
        for i in range(0, 4, 1):
            self.moveForward()
            time.sleep(0.25)
            self.moveBackward()
            time.sleep(0.25)
        self.hover()
        self._speed = ospeed

    def printDefault(self, *args):
        if os.name == 'posix':
            print(self._pDefaultStr,)
            if len(args) > 0:
                for i in args:
                    print( i,)
                print(self._pDefaultStr)

    def printRed(self, *args):
        if os.name == 'posix':
            print(self._pRedStr,)
            if len(args) > 0:
                for i in args:
                    print(i,)
                print(self._pDefaultStr)

    def printGreen(self, *args):
        if os.name == 'posix':
            print(self._pGreenStr,)
            if len(args) > 0:
                for i in args:
                    print(i,)
                print(self._pDefaultStr)

    def printYellow(self, *args):
        if os.name == 'posix':
            print(self._pYellowStr,)
            if len(args) > 0:
                for i in args:
                    print(i,)
                print(self._pDefaultStr)

    def printBlue(self, *args):
        if os.name == 'posix':
            print(self._pBlueStr,)
            if len(args) > 0:
                for i in args:
                    print(i,)
                print(self._pDefaultStr)

    def printPurple(self, *args):
        if os.name == 'posix':
            print(self._pPurpleStr,)
            if len(args) > 0:
                for i in args:
                    print(i,)
                print(self._pDefaultStr)

    def printLineUp(self):
        if os.name == 'posix':
            print(self._pLineUpStr,)


# =-
# Threads & Thread-Sidekicks ###=-
# =-
# Idea: the network thread listens to the given network-stream and communication-pipes of other processes, such as for video or navdata-decoding.
# 		In case the connection to the drone is cut off for more than 2 seconds (so no keep-alive-command has been sent) the network
#		needs to reconnect. In order to do so the (private) function "_netrecon" starts after 0.1 seconds of no incoming navdata-datapacket to
#		reconnect all given network-sockets.
    def _heartbeat(self):
        """
        If the drone does not get a command, it will mutter after 50ms (CTRL watchdog / state[28] will set to 1)
        and panic after 2 seconds and abort data-communication on port 5554 (then you have to initialize the network again).
        Heartbeat will reset the watchdog and, by the way, the ACK_BIT (state[6], to accept any other AT*CONFIG command)
        If mainthread isn't alive anymore (because program crashed or
        whatever), heartbeat will initiate the shutdown.
        """
        if str(threading.enumerate()).count("MainThread, stopped") or str(threading.enumerate()).count("MainThread") == 0:
            self.shutdown()
        else:
            self.at("COMWDG", [])

    # CheckAndReact is periodically called by the receiveData-Thread to check for mainly for critical status-error(s) and
    # changed debug-modes.
    def _checkAndReact(self, debug, showCommands):
        # Automatic process-commands, used for syncing debugging-bits to
        # child-processes
        if debug != self.debug:
            debug = self.debug
            if debug:
                self._NavData_pipe.send("debug")
                self._Video_pipe.send("debug")
            else:
                self._NavData_pipe.send("undebug")
                self._Video_pipe.send("undebug")
        if showCommands != self.showCommands:
            showCommands = self.showCommands
            if showCommands:
                self._NavData_pipe.send("showCommands")
                self._Video_pipe.send("showCommands")
            else:
                self._NavData_pipe.send("hideCommands")
                self._Video_pipe.send("hideCommands")
        # Communication problem, shutting down
        if self.stopOnComLoss and self._State[30]:
            self.shutdown()
            sys.exit()
        return (debug, showCommands)

    # Thread for sending the configuration. It is asynchronous but save.
    # The configuration-requests are in a queue, the first entry is sent. NavData will contain a "Control command ACK" status-bit,...
    # ...that configuration is ready to be set. This will be confirmed and the procedure waits until this bit is 0 again; then the next entry will be processed.
    # In savemode, there is a check whether the configuration has been changed
    # correctly by requesting the current/latest configuration and
    # double-checking this value.
    def _sendConfig(self):
        sleeptime, getconfigtag, self._sendConfigRunning = 0.001, False, True
        while not self._networksuicide:
            # If there is something in the queue...
            if len(self._ConfigQueue):
                if self._ConfigQueue[0][-1]:
                    # ...check for multiuserconfig-request (and send it)
                    self.sendConfigIDs()
                # Set tag, to show sending is in process
                self._ConfigSending = True
                qlen = len(self._ConfigQueue)
                # Testing for double entries, preventing a ping-pong in
                # save-mode
                if qlen > 1:
                    i = 1
                    while True:
                        if i >= qlen:
                            break
                        if self._ConfigQueue[0][0].lower() == self._ConfigQueue[i][0].lower():
                            # Delete double entries
                            self._ConfigQueue.remove(self._ConfigQueue[0])
                            qlen = len(self._ConfigQueue)
                        else:
                            i += 1
                # Send the first entry in queue
                self.at("CONFIG", self._ConfigQueue[0][:-1])
                getconfigtag, configconfirmed, configreconfirmed = False, False, False
                # Wait for confirmation-bit from drone...
                while not configconfirmed and not self._networksuicide:
                    if self._State[6] and not configreconfirmed and not self._networksuicide:
                        # ...and send reset the confirmation-bit
                        self.at("CTRL", [5, 0])
                        configreconfirmed = True
                    if not self._State[6] and configreconfirmed and not self._networksuicide:
                        # Wait for the reset of the confirmation-bit
                        configconfirmed = True
                    time.sleep(sleeptime)
                # It seems that the drone stores configurations not always
                # correctly; therfore, here is a save-mode:
                if self.sendConfigSaveMode and not self._networksuicide:
                    # Wait for the next configuration-list
                    lastConfigDataCount = self._ConfigDataCount
                    self.getConfig()
                    while lastConfigDataCount == self._ConfigDataCount and not self._networksuicide:
                        time.sleep(sleeptime)
                # New & Optimized
                    for i in range(0, len(self._ConfigData), 1):
                        if self._ConfigData[i][0].find(self._ConfigQueue[0][0]) > -1:
                            if self._ConfigData[i][1] != self._ConfigQueue[0][1]:
                                if self.debug:
                                    print("   Configuration missmatched, resending !")
                                    print("   " + self._ConfigData[i][0] + " should be \"" + self._ConfigQueue[0][1] + "\" is \"" + self._ConfigData[i][1] + "\"")
                                    # If value is not correctly set, requeue !
                                    self._ConfigQueue.append(
                                        self._ConfigQueue[0])
                # Configuration has been (correctly) set, delete request from
                # queue and go on
                self._ConfigQueue.remove(self._ConfigQueue[0])
                if self._networksuicide:
                    self._ConfigQueue = []
            if not len(self._ConfigQueue):
                if not getconfigtag:
                    self.getConfig()
                    getconfigtag = True
                    self._ConfigSending = False
                else:
                    time.sleep(sleeptime)
        if self.debug:
            print("sendConfig-Tread :   committed suicide")

    def _receiveData(self):
        self._net_pipes = []
        self._net_pipes.append(self._NavData_pipe)
        self._net_pipes.append(self._Video_pipe)
        self._Config_pipe = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)  # TCP
        self._Config_pipe.setblocking(0)
        self._Config_pipe.connect_ex((self.DroneIP, self.CTLPort))
        self._net_pipes.append(self._Config_pipe)
        VideoIsDead, configdata, cfgdata, cmd = False, [], "", ""
        vDecodeRunning, debug, showCommands, self._receiveDataRunning = False, False, False, True

        while not self._networksuicide:
            # When something is in a pipe...
            in_pipe, dummy1, dummy2 = select.select(
                self._net_pipes, [], [], 0.1)
            for ip in in_pipe:														# ...go and get it
                # Receiving sensor-values from NavData-process
                if ip == self._NavData_pipe:
                    self._NavData, self._State, self._NavDataCount, self._NavDataTimeStamp, self._NavDataDecodingTime, self._NoNavData = self._NavData_pipe.recv(
                    )
                # Receiving imagedata and feedback from videodecode-process
                if ip == self._vdecode_pipe:
                    # Imagedata
                    cmd, VideoImageCount, VideoImage, VideoDecodeTime = self._vdecode_pipe.recv()
                    if self.showCommands and cmd != "Image":
                        print("** vDec -> Com :", cmd)
                    if cmd == "suicided":
                        # videodecode-process died
                        self._Video_pipe.send("vd died")
                    if cmd == "foundCodec":
                        # the codec of the videostream has been found, do not
                        # flood anymore
                        self._Video_pipe.send("foundCodec")
                    if cmd == "VideoUp":
                        self._VideoReady = True				# Imagedata is available
                    if cmd == "keypressed":
                        self._vKey = VideoImage				# Pressed key on window
                    if cmd == "reset":
                        # proxy to videodecode-process
                        self._Video_pipe.send(cmd)
                    if cmd == "Image":															# Imagedata !
                        self._VideoImageCount = VideoImageCount
                        self._VideoImage = VideoImage
                        self._VideoDecodeTime = VideoDecodeTime
                        self._VideoDecodeTimeStamp = time.time() - \
                            self._startTime
                # Receiving feedback from videostream-process
                if ip == self._Video_pipe:
                    cmd = self._Video_pipe.recv()
                    if self.showCommands and cmd != "":
                        print("** Vid -> Com : ", cmd)
                    # videodecode-process should start
                    if cmd == "vDecProc":
                        if not vDecodeRunning:
                            try:
                                if self._vDecodeProcess == True:
                                    pass
                            except:
                                self._vDecodeProcess = multiprocessing.Process(
                                    target=vDecode, args=(self._VidPipePath, self._vdecodeChild_pipe, os.getpid()))
                            self._vDecodeProcess.start()
                            self._net_pipes.append(self._vdecode_pipe)
                            self._vDecodeRunning = True
                        self._Video_pipe.send("vDecProcON")
# else:
# self._vdecode_pipe.send(cmd)           # If / elif / else is somehow
# not working here...whyever
                    if cmd == "VideoDown":
                        # videodecode-process stopped
                        self._VideoReady = False
                    if cmd == "saveVideo":
                        # no preprocessing of the video
                        self._SaveVideo = True
                    if cmd == "unsaveVideo":
                        # preprocessing activated again
                        self._SaveVideo = False
                    if cmd == "debug":
                        # proxy to videodecode-process
                        self._vdecode_pipe.send(cmd)
                    if cmd == "showCommands":
                        # proxy to videodecode-process
                        self._vdecode_pipe.send(cmd)
                    if cmd == "hideCommands":
                        # proxy to videodecode-process
                        self._vdecode_pipe.send(cmd)
                    if cmd == "show":
                        # proxy to videodecode-process
                        self._vdecode_pipe.send(cmd)
                    if cmd == "hide":
                        # proxy to videodecode-process
                        self._vdecode_pipe.send(cmd)
                    if cmd == "vDecProcKill":
                        # videodecode-process should switch off
                        self._vdecode_pipe.send("die!")
                        vDecodeRunning = False
                # Receiving drone-configuration
                if ip == self._Config_pipe and not self._networksuicide:
                    try:
                        if self._networksuicide:
                            # Does not stop sometimes, so the loop will be
                            # forced to stop
                            break
                        # Data comes in two or three packages
                        cfgdata = cfgdata + self._Config_pipe.recv(65535)
                        # Last byte of sent config-file, everything was
                        # received
                        if cfgdata.count("\x00"):
                            if self._networksuicide:
                                break
                            # Split the huge package into a configuration-list
                            configdata = (cfgdata.split("\n"))
                            for i in range(0, len(configdata), 1):
                                # Split the single configuration-lines into
                                # configuration and value
                                configdata[i] = configdata[i].split(" = ")
                            # Last value is "\x00"
                            self._ConfigData = configdata[:-1]
                            # Set a timestamp for a better coordination
                            self._ConfigDataTimeStamp = time.time() - \
                                self._startTime
                            # Alters the count of received Configdata for a
                            # better coordination
                            self._ConfigDataCount += 1
                            configdata, cfgdata = [], ""
                            if self.showCommands:
                                print("Got " + str(len(self._ConfigData)) + " Configdata " + str(time.time() - self._calltime))
                            self._calltime = 0
                    except IOError:
                        pass
                # Check for errors and things to change
                debug, showCommands = self._checkAndReact(debug, showCommands)
        if self.debug:
            print("receiveData-Thread : committed suicide")

    def _stopnetwork(self):
        self._networksuicide = True

# =-
# Compatibility Commands ###=-
# =-
# While programming this API I changed some command-names
# This section converts the old commands into the new ones
    # Controls engines directly, overriding control loops.
    def pwm(self, fl, fr, rl, rr):
        if fl > 64000:
            fl = 64000
        if fr > 64000:
            fr = 64000
        if rl > 64000:
            rl = 64000
        if rr > 64000:
            rr = 64000
        self.at("PWM", [int(fl), int(fr), int(rr), int(rl)])

    def groundVideo(self, *args):
        self.groundCam(*args)

    def frontVideo(self, *args):
        self.frontCam(*args)

###############################################################################
# Internal Subfunctions
###############################################################################
    def _checkSpeedValue(self, value):
        try:
            speed = float(value)
            if self.valueCorrection:
                speed = max(-1.0, speed)
                speed = min(1.0, speed)
        except:
            speed = self._speed
        return speed

# Checks the inputs for the right length


def normalLen8(value):
    value, zero = str(value), "00000000"
    vlen = min(len(value), 8)
    normal = zero[0:8 - vlen] + value[0:8]
    return normal[0:8].lower()

##########################################################################
###### Receive and Decode Video																######
##########################################################################
# If the ps_drone-process has crashed, recognize it and kill yourself


def watchdogV(parentPID, ownPID):
    global commitsuicideV
    while not commitsuicideV:
        time.sleep(1)
        try:
            os.getpgid(parentPID)
        except:
            try:
                subprocess.Popen(["kill", str(
                    os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except:
                pass

# Thread to capture, decode and display the video-stream

def vCapture(VidPipePath, parent_pipe):
    import cv2
    global vCruns, commitsuicideV, showVid, lockV, debugV, vidPipePath

#    print("vCapture called!")
#    print("VidPipePath: "+VidPipePath+".")
    #vidPipePath = VidPipePath
    Drone.vidTemp = VidPipePath
    
#	cv2.startWindowThread()
    show = False
    hide = True
    vCruns = True
    t = time.time()
    parent_pipe.send(("VideoUp", 0, 0, 0))
    capture = cv2.VideoCapture(VidPipePath)
    #    print ("VidPipePath " + Drone.vidTemp)

    
    ImgCount = 0
    if debugV:
        print("CAPTURE: " + str(time.time() - t))
    time.sleep(0.1)
    parent_pipe.send(("foundCodec", 0, 0, 0))
    declag = time.time()
    count = -3
    imageXsize = 0
    imageYsize = 0
    windowName = "PS-Drone"
    codecOK = False
    lastKey = ""
    cc = 0

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    for i in range(50):
        generator = aruco.drawMarker(dictionary, i, 100)
        
    while not commitsuicideV:
        decTimeRev = time.time()
        receiveWatchdog = threading.Timer(2.0, VideoReceiveWatchdog, [
                                          parent_pipe, "vCapture", debugV])  # Resets video if something hangs
        receiveWatchdog.start()
        success, image = capture.read()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary)

        #
        im1 = cv2.imread('poket/1.png')
        im2 = cv2.imread('poket/2.png')
        im3 = cv2.imread('poket/3.png')
        im4 = cv2.imread('poket/4.png')
        im5 = cv2.imread('poket/5.png')
        im6 = cv2.imread('poket/6.png')
               
        if not corners:
            imagepaste = image
        else:
            if ids[0][0] == 0:
                image = Image.fromarray(np.uint8(image))
                im1 = Image.fromarray(np.uint8(im1))
                image.paste(im1, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
            if ids[0][0] == 1:
                image = Image.fromarray(np.uint8(image))
                im2 = Image.fromarray(np.uint8(im2))
                image.paste(im2, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
            if ids[0][0] == 2:
                image = Image.fromarray(np.uint8(image))
                im3 = Image.fromarray(np.uint8(im3))
                image.paste(im3, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
            if ids[0][0] == 3:
                image = Image.fromarray(np.uint8(image))
                im4 = Image.fromarray(np.uint8(im4))
                image.paste(im4, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
            if ids[0][0] == 4:
                image = Image.fromarray(np.uint8(image))
                im5 = Image.fromarray(np.uint8(im5))
                image.paste(im5, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
            if ids[0][0] == 5:
                image = Image.fromarray(np.uint8(image))
                im6 = Image.fromarray(np.uint8(im6))
                image.paste(im6, (corners[0][0][0][0].astype(int), corners[0][0][0][1].astype(int)))
                
        image = np.asarray(image)


 

        #

        cc += 1
        receiveWatchdog.cancel()
        decTime = decTimeRev - time.time()
        tlag = time.time() - declag

        if not codecOK:
            if image.shape[:2] == (360, 640) or image.shape[:2] == (368, 640) or image.shape[:2] == (720, 1280) or image.shape[:2] == (1080, 1920):
                codecOK = True
                if debugV:
                    print("Codec seems OK")
            else:
                if debugV:
                    print("Codec failure")
                parent_pipe.send(("reset", 0, 0, 0))
                commitsuicideV = True
        if codecOK:
            if not (imageXsize == image.shape[1]) or not (imageYsize == image.shape[0]):
                cv2.destroyAllWindows()
                imageYsize, imageXsize = image.shape[:2]
                windowName = "PS-Drone - " + \
                    str(imageXsize) + "x" + str(imageYsize)
            if success:
                if tlag > 0.02:
                    count += 1
                if count > 0:
                    ImgCount += 1
                    if not show and not hide:
                        cv2.destroyAllWindows()
                        hide = True
                    if show:
                        cv2.imshow(windowName, image)
                        key = cv2.waitKey(1)
                        if key > -1:
                            parent_pipe.send(
                                ("keypressed", 0, chr(key % 256), 0))
                    parent_pipe.send(("Image", ImgCount, image, decTime))
            else:
                time.sleep(0.01)
            declag = time.time()

            if showVid:
                if not show:
                    show = True
                    cv2.destroyAllWindows()
            else:
                if show:
                    show = False
                    cv2.destroyAllWindows()
    vCruns = False
    cv2.destroyAllWindows()
    capture.release()
    if debugV:
        print("vCapture-Thread :    committed suicide")

# Process to decode the videostream in the FIFO-Pipe, stored there from main-loop.
# Storing and decoding must not be processed in the same process, thats why decoding is external.
# vDecode controls the vCapture-thread which captures and decodes finally
# the videostream.


def vDecode(VidPipePath, parent_pipe, parentPID):
    global vCruns, commitsuicideV, showVid, lockV, debugV
    showCommands = False
    Thread_vCapture = threading.Thread(
        target=vCapture, args=(VidPipePath, parent_pipe))
    Thread_vCapture.start()
    Thread_watchdogV = threading.Thread(
        target=watchdogV, args=[parentPID, os.getpid()])
    Thread_watchdogV.start()

    while not commitsuicideV:
        # When something is in a pipe...
        in_pipe, out_pipe, dummy2 = select.select([parent_pipe], [], [], 0.1)
        cmd = parent_pipe.recv()
        if showCommands:
            print("** Com -> vDec : ", cmd)
        if cmd == "die!":
            commitsuicideV = True
        elif cmd == "reset":
            commitsuicideV = True
        elif cmd == "show":
            showVid = True
        elif cmd == "hide":
            showVid = False
        elif cmd == "debug":
            debugV = True
            print("vDecode-Process :    running")
            if vCruns:
                print("vCapture-Thread :    running")
        elif cmd == "undebug":
            debugV = False
        elif cmd == "showCommands":
            showCommands = True
        elif cmd == "hideCommands":
            showCommands = False
    Thread_vCapture.join()
    parent_pipe.send(("suicided", 0, 0, 0))
    time.sleep(0.1)
    if debugV:
        print("vDecode-Process :    committed suicide")

#####################################################


def VideoReceiveWatchdog(parent_pipe, name, debugV):
    if debugV:
        print("WHATCHDOG reset von", name)
    parent_pipe.send(("reset", 0, 0, 0))


def mainloopV(DroneIP, VideoPort, VidPipePath, parent_pipe, parentPID):
    inited, preinited, suicide, debugV, showCommands, slowVideo = False, False, 0, False, False, False
    rawVideoFrame, VidStreamSnippet, VidStreamSnippetAvalible, iFrame, FrameCount = "", "", False, False, 0
    saveVideo, unsureMode, searchCodecTime, frameRepeat, burstFrameCount = False, True, 0, 1, 0
    reset, resetCount, commitsuicideV, foundCodec = False, 0, False, False

    vstream_pipe, pipes = None, [parent_pipe]
    vdecode_pipe, vdecode_childpipe = multiprocessing.Pipe()
    pipes.append(vdecode_pipe)
    Thread_watchdogV = threading.Thread(
        target=watchdogV, args=[parentPID, os.getpid()])
    Thread_watchdogV.start()

    while not commitsuicideV:
        # When something is in a pipe...
        in_pipe, out_pipe, dummy2 = select.select(pipes, [], [], 0.1)
        for ip in in_pipe:
            if ip == parent_pipe:
                cmd = parent_pipe.recv()
                if showCommands:
                    print("** Com -> Vid : ", cmd)
                if cmd == "die!":
                    if inited:
                        suicide = True
                        parent_pipe.send("vDecProcKill")
                        dummy = 0
                    else:
                        commitsuicideV = True
                elif cmd == "foundCodec":
                    foundCodec = True
                elif cmd == "reset" and not reset:  # and resetCount<3:
                    inited, preinited, foundCodec = False, False, False
                    rawVideoFrame, VidStreamSnippet = "", ""
                    VidStreamSnippetAvalible = False
                    iFrame, FrameCount, reset = False, 0, True
                    unsureMode, searchCodecTime = True, 0
                    burstFrameCount = 0
                    resetCount += 1
                    parent_pipe.send("vDecProcKill")
                elif cmd == "slowVideo":
                    slowVideo = True
                    frameRepeat = 1
                elif cmd == "midVideo":
                    slowVideo = True
                    frameRepeat = 4
                elif cmd == "fastVideo":
                    slowVideo = False
                    frameRepeat = 1
                elif cmd == "saveVideo":
                    saveVideo = True
                    parent_pipe.send("saveVideo")
                elif cmd == "unsaveVideo":
                    saveVideo = False
                    parent_pipe.send("unsaveVideo")
                elif cmd == "showCommands":
                    showCommands = True
                    parent_pipe.send("showCommands")
                elif cmd == "hideCommands":
                    showCommands = False
                    parent_pipe.send("hideCommands")
                elif cmd == "debug":
                    debugV = True
                    print("Video-Process :      running")
                    parent_pipe.send("debug")
                elif cmd == "undebug":
                    debugV = False
                    parent_pipe.send("undebug")
                elif cmd == "init" and not inited and not preinited:
                    preinited = True
                    try:
                        os.mkfifo(VidPipePath)
                    except:
                        pass
                    parent_pipe.send("vDecProc")
                elif cmd == "vDecProcON":
                    rawVideoFrame = ""
                    VidStreamSnippet = ""
                    iFrame = False
                    FrameCount = 0
                    foundCodec = False
                    searchCodecTime = 0
                    if not vstream_pipe:
                        vstream_pipe = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
                        vstream_pipe.setblocking(0)
                        vstream_pipe.connect_ex((DroneIP, VideoPort))
                        pipes.append(vstream_pipe)
                    write2pipe = open(VidPipePath, "w+")
                    suicide = False
                    inited = True
                    preinited = False
                    unsureMode = True
                elif cmd == "uninit" and inited:
                    parent_pipe.send("vDecProcKill")
                elif cmd == "vd died":
                    if inited and not reset:
                        pipes.remove(vstream_pipe)
                        vstream_pipe.shutdown(socket.SHUT_RDWR)
                        vstream_pipe.close()
                        write2pipe.close()
                        inited = False
                        if suicide:
                            commitsuicideV = True
                        parent_pipe.send("VideoDown")
                    try:
                        os.remove(VidPipePath)
                    except:
                        pass
                    if not inited and reset:
                        try:
                            os.mkfifo(VidPipePath)
                        except:
                            pass
                        parent_pipe.send("VideoDown")
                        parent_pipe.send("vDecProc")
                        parent_pipe.send("debug")
                        reset = False
                        burstFrameCount = 0
                else:
                    parent_pipe.send(cmd)

            # Grabs the Videostream and store it in a fifo-pipe for decoding.
            # The decoder has to guess the videostream-format which takes around 266 video-frames.
            #    So the stream is preprocessed, I-Frames will cut out while initiation and a flood of copies
            #	 will be send to the decoder, till the proper decoder for the videostream is found.
            # In case of a slow or midspeed-video, only a single or a few
            # copied I-frames are sent to the decoder.
            if ip == vstream_pipe:
                receiveWatchdog = threading.Timer(2.0, VideoReceiveWatchdog, [
                                                  parent_pipe, "Video Mainloop", debugV, ])  # Resets video if something hangs
                receiveWatchdog.start()
                videoPackage = vstream_pipe.recv(65535)
                receiveWatchdog.cancel()
                if len(videoPackage) == 0:
                    commitsuicideV = True
                else:
                    if inited and not reset:
                        # An MPEG4-Stream is not confirmed, fallback to
                        # savemode ?
                        if unsureMode:
                            # Video is freshly initiated
                            if not searchCodecTime and not len(VidStreamSnippet):
                                searchCodecTime = time.time()
                            # Collecting VidStreamSnipped for later use
                            if (time.time() - searchCodecTime) < 0.15:
                                VidStreamSnippet += videoPackage
                            # Waited too long for an MPEG4 stream
                            # confirmation...
                            if (time.time() - searchCodecTime) > 2.0:
                                # ... fall back to savemode
                                saveVideo = True
                                # Inform the main process
                                parent_pipe.send("saveVideo")
                                unsureMode = False
                                # switch off codec guess speed-up
                                foundCodec = True
                        if not saveVideo:
                            #						if len(videoPackage) == 0:		commitsuicideV = True
                            #						else:
                            # Found a new MPEG4-Frame
                            if videoPackage[31:40].find("\x00\x00\x00") > 3:
                                FrameCount += 1
                                # Processing the last frame
                                # If the last frame was an I-frame
                                if iFrame:
                                    # ... save it as VideoStreamSnippet for later use
                                    VidStreamSnippet = rawVideoFrame
                                    # OpenCV guessed the used Codec
                                    if foundCodec:
                                        # Send just the iFrame (openCV stores
                                        # about 5 in its queue),
                                        if slowVideo:
                                            # ... so repeat for less delay in midVideo()-mode
                                            for i in range(0, frameRepeat, 1):
                                                write2pipe.write(
                                                    VidStreamSnippet)
                                    iFrame = False
                                else:
                                    pass
                                if not slowVideo:								# For all last Frames
                                    if foundCodec:
                                        try:
                                            write2pipe.write(rawVideoFrame)
                                        except:
                                            pass
                                # Flood the pipe with the last iFrames, so that
                                # openCV can guess the codec faster
                                if not foundCodec:
                                    for i in range(0, 5):
                                        try:
                                            write2pipe.write(rawVideoFrame)
                                            burstFrameCount += 1
                                        except:
                                            pass
                                # Processing new Frames
                                # Found an I-Frame
                                if ord(videoPackage[30]) == 1:
                                    # Delete the data previous to the first
                                    # iFrame
                                    rawVideoFrame = ""
                                    unsureMode, iFrame = False, True
                                # Found a P-Frame
                                elif ord(videoPackage[30]) == 3:
                                    unsureMode = False
                                else:  # Found an odd h264-frametype
                                    if debugV:
                                        print("*** Odd h264 Frametype: ", FrameCount,)
                                        for i in range(31, 43, 1):
                                            print(ord(videoPackage[i]),)
                                        print(" - ", videoPackage[31:40].find("\x00\x00\x00"), ord(videoPackage[30]))
                                rawVideoFrame = ""
                            # Collecting data for the next frame from stream
                            rawVideoFrame += videoPackage
                        else:  # (saveVideo-Mode)
                            if foundCodec:
                                write2pipe.write(videoPackage)
                            else:
                                for i in range(0, 2):
                                    write2pipe.write(VidStreamSnippet)
                                    burstFrameCount += 1
                        if not foundCodec and burstFrameCount > 350:
                            parent_pipe.send(("reset", 0, 0, 0))
                            burstFrameCount = 0
                            if debugV:
                                print("To many pictures send while guessing the codec. Resetting.")

    try:
        vstream_pipe.shutdown(socket.SHUT_RDWR)
        vstream_pipe.close()
    except:
        pass
    try:
        write2pipe.close()
    except:
        pass
    try:
        vstream_pipe.close()
    except:
        pass
    try:
        VidPipe = open(VidPipePath, "r")
        r = "1"
        while len(r):
            r = VidPipe.read()
        FIFO.close()
    except:
        pass
    try:
        os.remove(VidPipePath)
    except:
        pass
    if debugV:
        print("Video-Process :      committed suicide")


##########################################################################
###### Receive and Decode NavData															######
##########################################################################
# Description:
# It follows lousy code for abetter documentation! Later there will be lousy code because of laziness; I will correct it later....maybe.
# You will (normally) find the names of the official AR.drone SDK 2.0, some comments and the official data type of that value.
# A lot of entries are reversed engineered; for some, I have no idea what they are doing or what their meaning is.
# It would be nice if you could give me a hint if you have some further
# information.

##### Header ##################################################################
def decode_Header(data):
    # Bit 00-07: FLY_MASK, VIDEO_MASK, VISION_MASK, CONTROL_MASK, ALTITUDE_MASK, USER_FEEDBACK_START, COMMAND_MASK, CAMERA_MASK
    # Bit 08-15: TRAVELLING_MASK, USB_MASK, NAVDATA_DEMO_MASK, NAVDATA_BOOTSTRAP, MOTORS_MASK, COM_LOST_MASK, SOFTWARE_FAULT, VBAT_LOW
    # Bit 16-23: USER_EL, TIMER_ELAPSED, MAGNETO_NEEDS_CALIB, ANGLES_OUT_OF_RANGE, WIND_MASK, ULTRASOUND_MASK, CUTOUT_MASK, PIC_VERSION_MASK
    # Bit 24-31: ATCODEC_THREAD_ON, NAVDATA_THREAD_ON, VIDEO_THREAD_ON,
    # ACQ_THREAD_ON, CTRL_WATCHDOG_MASK, ADC_WATCHDOG_MASK, COM_WATCHDOG_MASK,
    # EMERGENCY_MASK
    stateBit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 0: FLY MASK :					(0) ardrone is landed, (1) ardrone is flying
    stateBit[0] = data[1] & 1
    # 1: VIDEO MASK :					(0) video disable, (1) video enable
    stateBit[1] = data[1] >> 1 & 1
    # 2: VISION MASK :					(0) vision disable, (1) vision enable
    stateBit[2] = data[1] >> 2 & 1
    # 3: CONTROL ALGO :				(0) euler angles control, (1) angular speed control
    stateBit[3] = data[1] >> 3 & 1
    # 4: ALTITUDE CONTROL ALGO :              (0) altitude control inactive
    # (1) altitude control active
    stateBit[4] = data[1] >> 4 & 1
    stateBit[5] = data[1] >> 5 & 1  # 5: USER feedback : 				Start button state
    # 6: Control command ACK : 		(0) None, (1) one received
    stateBit[6] = data[1] >> 6 & 1
    # 7: CAMERA MASK : 				(0) camera not ready, (1) Camera ready
    stateBit[7] = data[1] >> 7 & 1
    # 8: Travelling mask : 			(0) disable, (1) enable
    stateBit[8] = data[1] >> 8 & 1
    # 9: USB key : 					(0) usb key not ready, (1) usb key ready
    stateBit[9] = data[1] >> 9 & 1
    # 10: Navdata demo : 				(0) All navdata, (1) only navdata demo
    stateBit[10] = data[1] >> 10 & 1
    # 11: Navdata bootstrap :                         (0) options sent in all
    # or demo mode, (1) no navdata options sent
    stateBit[11] = data[1] >> 11 & 1
    # 12: Motors status : 				(0) Ok, (1) Motors problem
    stateBit[12] = data[1] >> 12 & 1
    # 13: Communication Lost : 			(0) Com is ok, (1) com problem
    stateBit[13] = data[1] >> 13 & 1
    # 14: Software fault detected - user should land as quick as possible (1)
    stateBit[14] = data[1] >> 14 & 1
    stateBit[15] = data[1] >> 15 & 1  # 15: VBat low : 					(0) Ok, (1) too low
    # 16: User Emergency Landing :		(0) User EL is OFF, (1) User EL is ON
    stateBit[16] = data[1] >> 16 & 1
    # 17: Timer elapsed : 				(0) not elapsed, (1) elapsed
    stateBit[17] = data[1] >> 17 & 1
    # 18: Magnetometer calib state :  (0) Ok, no calibration needed, (1) not
    # ok, calibration needed
    stateBit[18] = data[1] >> 18 & 1
    # 19: Angles :						(0) Ok,	(1) out of range
    stateBit[19] = data[1] >> 19 & 1
    # 20: WIND MASK:					(0) Ok, (1) Too much wind
    stateBit[20] = data[1] >> 20 & 1
    # 21: Ultrasonic sensor :			(0) Ok, (1) deaf
    stateBit[21] = data[1] >> 21 & 1
    # 22: Cutout system detection :		(0) Not detected, (1) detected
    stateBit[22] = data[1] >> 22 & 1
    # 23: PIC Version number OK :             (0) a bad version number, (1)
    # version number is OK
    stateBit[23] = data[1] >> 23 & 1
    # 24: ATCodec thread ON : 			(0) thread OFF, (1) thread ON
    stateBit[24] = data[1] >> 24 & 1
    # 25: Navdata thread ON : 			(0) thread OFF, (1) thread ON
    stateBit[25] = data[1] >> 25 & 1
    # 26: Video thread ON : 			(0) thread OFF, (1) thread ON
    stateBit[26] = data[1] >> 26 & 1
    # 27: Acquisition thread ON : 		(0) thread OFF, (1) thread ON
    stateBit[27] = data[1] >> 27 & 1
    # 28: CTRL watchdog :                             (0) control is well
    # scheduled, (1) delay in control execution (> 5ms)
    stateBit[28] = data[1] >> 28 & 1
    # 29: ADC Watchdog :				(0) uart2 is good, (1) delay in uart2 dsr (> 5ms)
    stateBit[29] = data[1] >> 29 & 1
    # 30: Communication Watchdog :		(0) Com is ok, (1) com problem
    stateBit[30] = data[1] >> 30 & 1
    # 31: Emergency landing : 			(0) no emergency, (1) emergency
    stateBit[31] = data[1] >> 31 & 1
    stateBit[32] = data[2]
    stateBit[33] = data[3]
    # Alternative code:
    #	for i in range (0,32,1):	arState[i]=data>>i&1
    return (stateBit)

##### ID = 0 ### "demo" #######################################################


def decode_ID0(packet):		# NAVDATA_DEMO_TAG
    dataset = struct.unpack_from(
        "HHIIfffifffIffffffffffffIIffffffffffff", packet, 0)
    if dataset[1] != 148:
        print("*** ERROR : Navdata-Demo-Options-Package (ID=0) has the wrong size !!!")
    demo = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, [0, 0, 0], 0, [0, 0, 0], 0, [
        0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0], 0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0]]
    demo[0][0] = dataset[2] >> 15 & 1  # DEFAULT			(bool)
    demo[0][1] = dataset[2] >> 16 & 1  # INIT				(bool)
    demo[0][2] = dataset[2] >> 17 & 1  # LANDED			(bool)
    demo[0][3] = dataset[2] >> 18 & 1  # FLYING			(bool)
    # HOVERING			(bool)  (Seems like landing)
    demo[0][4] = dataset[2] >> 19 & 1
    demo[0][5] = dataset[2] >> 20 & 1  # TEST				(bool)
    demo[0][6] = dataset[2] >> 21 & 1  # TRANS_TAKEOFF		(bool)
    demo[0][7] = dataset[2] >> 22 & 1  # TRANS_GOFIX		(bool)
    demo[0][8] = dataset[2] >> 23 & 1  # TRANS_LANDING		(bool)
    demo[0][9] = dataset[2] >> 24 & 1  # TRANS_LOOPING		(bool)
    demo[0][10] = dataset[2] >> 25 & 1  # TRANS_NO_VISION	(bool)
    demo[0][11] = dataset[2] >> 26 & 1  # NUM_STATE			(bool)
    # vbat_flying_percentage	battery voltage (filtered) in percent	(uint32)
    demo[1] = dataset[3]
    # theta						pitch in degrees						(float)
    demo[2][0] = dataset[4] / 1000.0
    demo[2][1] = dataset[5] / 1000.0  # phi						roll  in degrees						(float)
    demo[2][2] = dataset[6] / 1000.0  # psi						yaw   in degrees						(float)
    # altitude					altitude in centimetres					(int32)
    demo[3] = dataset[7] / 10.0
    demo[4][0] = dataset[8]			# vx						estimated speed in X in mm/s			(float)
    demo[4][1] = dataset[9]			# vy						estimated speed in Y in mm/s			(float)
    demo[4][2] = dataset[10]		# vz						estimated speed in Z in mm/s			(float)
    # num_frames                              streamed frame index
    # (uint32) (Not used to integrate in video stage)
    demo[5] = dataset[11]
    for i in range(0, 9, 1):
        # detection_camera_rot            Camera parameters compute by
        # detection  (float matrix33)
        demo[6][i] = dataset[12 + i]
    for i in range(0, 3, 1):
        # detection_camera_trans	Deprecated ! Don't use !				(float vector31)
        demo[7][i] = dataset[21 + i]
    # detection_tag_index		Deprecated ! Don't use !				(uint32)
    demo[8] = dataset[24]
    # detection_camera_type   	Type of tag								(uint32)
    demo[9] = dataset[25]
    for i in range(0, 9, 1):
        # drone_camera_rot                        Camera parameters computed by
        # drone             (float matrix33)
        demo[10][i] = dataset[26 + i]
    for i in range(0, 3, 1):
        # drone_camera_trans		Deprecated ! Don't use !				(float vector31)
        demo[11][i] = dataset[35 + i]
    return(demo)

##### ID = 1 ### "time" #######################################################


def decode_ID1(packet):  # NAVDATA_TIME_TAG
    dataset = struct.unpack_from("HHI", packet, 0)
    if dataset[1] != 8:
        print("*** ERROR : navdata-time-Options-Package (ID=1) has the wrong size !!!")
    time = [0.0]
    # Value: 11 most significant bits represent the seconds, and the 21 least
    # significant bits represent the microseconds.
    for i in range(0, 21, 1):
        # Calculating the millisecond-part
        time[0] += ((dataset[2] >> i & 1) * (2 ** i))
    time[0] /= 1000000
    for i in range(21, 32, 1):
        # Calculating second-part
        time[0] += (dataset[2] >> i & 1) * (2 ** (i - 21))
    return(time)

##### ID = 2 ### "raw_measures" ##########################################


def decode_ID2(packet):  # NAVDATA_RAW_MEASURES_TAG
    dataset = struct.unpack_from("HHHHHhhhhhIHHHHHHHHHHHHhh", packet, 0)
    if dataset[1] != 52:
        print("*** ERROR : navdata-raw_measures-Options-Package (ID=2) has the wrong size !!!")
    raw_measures = [
        [0, 0, 0], [0, 0, 0], [0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 3, 1):
        # raw_accs[xyz]			filtered accelerometer-datas [LSB]	(uint16)
        raw_measures[0][i] = dataset[2 + i]
    for i in range(0, 3, 1):
        # raw_gyros[xyz]		filtered gyrometer-datas [LSB]		(int16)
        raw_measures[1][i] = dataset[5 + i]
    for i in range(0, 2, 1):
        # raw_gyros_110[xy]		gyrometers  x/y 110 deg/s [LSB]		(int16)
        raw_measures[2][i] = dataset[8 + i]
    # vbat_raw				battery voltage raw (mV)			(uint)
    raw_measures[3] = dataset[10]
    raw_measures[4] = dataset[11]		# us_debut_echo			[LSB]								(uint16)
    raw_measures[5] = dataset[12]		# us_fin_echo			[LSB]								(uint16)
    raw_measures[6] = dataset[13]		# us_association_echo	[LSB]								(uint16)
    raw_measures[7] = dataset[14]		# us_distance_echo		[LSB]								(uint16)
    raw_measures[8] = dataset[15]		# us_courbe_temps		[LSB]								(uint16)
    raw_measures[9] = dataset[16]		# us_courbe_valeur		[LSB]								(uint16)
    raw_measures[10] = dataset[17]		# us_courbe_ref			[LSB]								(uint16)
    raw_measures[11] = dataset[18]		# flag_echo_ini			[LSB]								(uint16)
    raw_measures[12] = dataset[19]		# nb_echo				[LSB]								(uint16)
    # sum_echo				juRef_st lower 16Bit, upper 16Bit=tags?	(uint32)
    raw_measures[13] = dataset[21]
    # alt_temp_raw			in Milimeter	(just lower 16Bit)	(int32)
    raw_measures[14] = dataset[23]
    raw_measures[15] = dataset[24]		# gradient				[LSB]								(int16)
    return(raw_measures)

##### ID = 3 ### "phys_measures" ##############################################


def decode_ID3(packet):  # NAVDATA_PHYS_MEASURES_TAG
    dataset = struct.unpack_from("HHfHffffffIII", packet, 0)
    if dataset[1] != 46:
        print("*** ERROR : navdata-phys_measures-Options-Package (ID=3) has the wrong size !!!")
    phys_measures = [0, 0, [0, 0, 0], [0, 0, 0], 0, 0, 0]
    phys_measures[0] = dataset[2]  # float32   accs_temp
    phys_measures[1] = dataset[3]  # uint16    gyro_temp
    # uint32    alim3V3              3.3volt alim [LSB]
    phys_measures[4] = dataset[10]
    # uint32    vrefEpson            ref volt Epson gyro [LSB]
    phys_measures[5] = dataset[11]
    # uint32    vrefIDG              ref volt IDG gyro [LSB]
    phys_measures[6] = dataset[12]
    # switch from little to big-endian
    dataset = struct.unpack_from(">HHfHffffffIII", packet, 0)
    for i in range(0, 3, 1):
        phys_measures[2][i] = dataset[4 + i]  # float32   phys_accs[xyz]
    for i in range(0, 3, 1):
        phys_measures[3][i] = dataset[7 + i]  # float32   phys_gyros[xyz]
    return(phys_measures)

##### ID = 4 ### "gyros_offsets" ##############################################


def decode_ID4(packet):  # NNAVDATA_GYROS_OFFSETS_TAG
    dataset = struct.unpack_from("HHfff", packet, 0)
    if dataset[1] != 16:
        print("*** ERROR : navdata-gyros_offsets-Options-Package (ID=4) has the wrong size !!!")
    gyros_offsets = [0, 0, 0]
    for i in range(0, 3, 1):
        # offset_g[xyz]				in deg/s					(float)
        gyros_offsets[i] = dataset[i + 2]
    return(gyros_offsets)

##### ID = 5 ### "euler_angles" ###############################################


def decode_ID5(packet):  # NAVDATA_EULER_ANGLES_TAG
    dataset = struct.unpack_from("HHff", packet, 0)
    if dataset[1] != 12:
        print("*** ERROR : navdata-euler_angles-Options-Package (ID=5) has the wrong size !!!")
    euler_angles = [0, 0]
    euler_angles[0] = dataset[2]  # float32   theta_a (head/back)
    euler_angles[1] = dataset[3]  # float32   phi_a   (sides)
    return(euler_angles)

##### ID = 6 ### "references" #################################################


def decode_ID6(packet):  # NAVDATA_REFERENCES_TAG
    dataset = struct.unpack_from("HHiiiiiiiiffffffIfffffI", packet, 0)
    if dataset[1] != 88:
        print("*** ERROR : navdata-references-Options-Package (ID=6) has the wrong size !!!")
    references = [[0, 0, 0], [0, 0], [0, 0, 0], [0.0, 0.0],
                  [0.0, 0.0], [0.0, 0.0], 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0]]
    # ref_theta  	Theta_ref_embedded [milli-deg]	(int32)
    references[0][0] = dataset[2]
    # ref_phi		Phi_ref_embedded [milli-deg]	(int32)
    references[0][1] = dataset[3]
    # ref_psi		Psi_ref_embedded [milli-deg]	(int32)
    references[0][2] = dataset[9]
    # ref_theta_I	Theta_ref_int [milli-deg]		(int32)
    references[1][0] = dataset[4]
    # ref_phi_I		Phi_ref_int [milli-deg]			(int32)
    references[1][1] = dataset[5]
    # ref_pitch		Pitch_ref_embedded [milli-deg]	(int32)
    references[2][0] = dataset[6]
    # ref_roll		Roll_ref_embedded [milli-deg]	(int32)
    references[2][1] = dataset[7]
    # ref_yaw		Yaw_ref_embedded [milli-deg/s]	(int32)
    references[2][2] = dataset[8]
    references[3][0] = dataset[10]  # vx_ref			Vx_Ref_[mm/s]					(float)
    references[3][1] = dataset[11]  # vy_ref			Vy_Ref_[mm/s]					(float)
    # theta_mod		Theta_modele [radian]			(float)
    references[4][0] = dataset[12]
    references[4][1] = dataset[13]  # phi_mod		Phi_modele [radian]				(float)
    references[5][0] = dataset[14]  # k_v_x											(float)
    references[5][1] = dataset[15]  # k_v_y											(float)
    references[6] = dataset[16]  # k_mode											(uint32)
    references[7][0] = dataset[17]  # ui_time										(float)
    references[7][1] = dataset[18]  # ui_theta										(float)
    references[7][2] = dataset[19]  # ui_phi											(float)
    references[7][3] = dataset[20]  # ui_psi											(float)
    references[7][4] = dataset[21]  # ui_psi_accuracy								(float)
    references[7][5] = dataset[22]  # ui_seq											(int32)
    return(references)

##### ID = 7 ### "trims" ######################################################


def decode_ID7(packet):  # NAVDATA_TRIMS_TAG
    dataset = struct.unpack_from("HHfff", packet, 0)
    if dataset[1] != 16:
        print("*** ERROR : navdata-trims-Options-Package (ID=7) has the wrong size !!!")
    trims = [0, 0, 0]
    trims[0] = dataset[2]  # angular_rates_trim									(float)
    trims[1] = dataset[3]  # euler_angles_trim_theta	[milli-deg]					(float)
    trims[2] = dataset[4]  # euler_angles_trim_phi	[milli-deg]					(float)
    return(trims)

##### ID = 8 ### "rc_references" ##############################################


def decode_ID8(packet):  # NAVDATA_RC_REFERENCES_TAG
    dataset = struct.unpack_from("HHiiiii", packet, 0)
    if dataset[1] != 24:
        print("*** ERROR : navdata-rc_references-Options-Package (ID=8) has the wrong size !!!")
    rc_references = [0, 0, 0, 0, 0]
    rc_references[0] = dataset[2]  # rc_ref_pitch		Pitch_rc_embedded			(int32)
    rc_references[1] = dataset[3]  # rc_ref_roll		Roll_rc_embedded			(int32)
    rc_references[2] = dataset[4]  # rc_ref_yaw		Yaw_rc_embedded				(int32)
    rc_references[3] = dataset[5]  # rc_ref_gaz		Gaz_rc_embedded				(int32)
    rc_references[4] = dataset[6]  # rc_ref_ag		Ag_rc_embedded				(int32)
    return(rc_references)

##### ID = 9 ### "pwm" ########################################################


def decode_ID9(packet):  # NAVDATA_PWM_TAG
    dataset = struct.unpack_from("HHBBBBBBBBffffiiifiiifHHHHff", packet, 0)
    if dataset[1] != 76 and dataset[1] != 92:  # 92 since firmware 2.4.8 ?
        print("*** ERROR : navdata-navdata_pwm-Options-Package (ID=9) has the wrong size !!!")
        # print("Soll: 76     Ist:",dataset[1])
    pwm = [[0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0, 0.0, 0.0,
           [0, 0, 0], 0.0, [0, 0, 0, 0.0], [0, 0, 0, 0], 0.0, 0.0]
    for i in range(0, 4, 1):
        pwm[0][i] = dataset[2 + i]  # motor1/2/3/4		[Pulse-width mod]	(uint8)
    for i in range(0, 4, 1):
        # sat_motor1/2/3/4	[Pulse-width mod]	(uint8)
        pwm[1][i] = dataset[6 + i]
    pwm[2] = dataset[10]			# gaz_feed_forward		[Pulse-width mod]	(float)
    pwm[3] = dataset[11]			# gaz_altitud			[Pulse-width mod]	(float)
    pwm[4] = dataset[12]			# altitude_integral		[mm/s]				(float)
    pwm[5] = dataset[13]			# vz_ref				[mm/s]				(float)
    pwm[6][0] = dataset[14]			# u_pitch				[Pulse-width mod]	(int32)
    pwm[6][1] = dataset[15]			# u_roll				[Pulse-width mod]	(int32)
    pwm[6][2] = dataset[16]			# u_yaw					[Pulse-width mod]	(int32)
    pwm[7] = dataset[17]			# yaw_u_I				[Pulse-width mod]	(float)
    pwm[8][0] = dataset[18]			# u_pitch_planif		[Pulse-width mod]	(int32)
    pwm[8][1] = dataset[19]			# u_roll_planif			[Pulse-width mod]	(int32)
    pwm[8][2] = dataset[20]			# u_yaw_planif			[Pulse-width mod]	(int32)
    pwm[8][3] = dataset[21]			# u_gaz_planif			[Pulse-width mod]	(float)
    for i in range(0, 4, 1):
        pwm[9][i] = dataset[22 + i]  # current_motor1/2/3/4	[mA]				(uint16)
    pwm[10] = dataset[26]			# altitude_prop			[Pulse-width mod]	(float)
    pwm[11] = dataset[27]			# altitude_der			[Pulse-width mod]	(float)
    return(pwm)

##### ID = 10 ### "altitude" #############################################


def decode_ID10(packet):  # NAVDATA_ALTITUDE_TAG
    dataset = struct.unpack_from("HHifiiffiiiIffI", packet, 0)
    if dataset[1] != 56:
        print("*** ERROR : navdata-navdata_altitude-Options-Package (ID=10) has the wrong size !!!")
    altitude = [0, 0.0, 0, 0, 0.0, 0.0, [0, 0, 0], 0, [0, 0], 0]
    altitude[0] = dataset[2]			# altitude_vision	[mm]					(int32)
    altitude[1] = dataset[3]			# altitude_vz		[mm/s]					(float)
    altitude[2] = dataset[4]			# altitude_ref		[mm]					(int32)
    altitude[3] = dataset[5]			# altitude_raw		[mm]					(int32)
    altitude[4] = dataset[6]			# obs_accZ			Observer AccZ [m/s2]	(float)
    altitude[5] = dataset[7]			# obs_alt			Observer altitude US [m](float)
    for i in range(0, 3, 1):
        altitude[6][i] = dataset[8 + i]  # obs_x				3-Vector				(int32)
    altitude[7] = dataset[11]			# obs_state			Observer state [-]		(uint32)
    for i in range(0, 2, 1):
        altitude[8][i] = dataset[12 + i]  # est_vb			2-Vector				(float)
    altitude[9] = dataset[14]			# est_state			Observer flight state 	(uint32)
    return(altitude)

##### ID = 11 ### "vision_raw" ###########################################


def decode_ID11(packet):  # NAVDATA_VISION_RAW_TAG
    dataset = struct.unpack_from("HHfff", packet, 0)
    if dataset[1] != 16:
        print("*** ERROR : navdata-vision_raw-Options-Package (ID=11) has the wrong size !!!")
    vision_raw = [0, 0, 0]
    for i in range(0, 3, 1):
        vision_raw[i] = dataset[2 + i]  # vision_tx_raw (xyz)				(float)
    return(vision_raw)

##### ID = 12 ### "vision_of" #################################################


def decode_ID12(packet):  # NAVDATA_VISION_OF_TAG
    dataset = struct.unpack_from("HHffffffffff", packet, 0)
    if dataset[1] != 44:
        print("*** ERROR : navdata-vision_of-Options-Package (ID=12) has the wrong size !!!")
    vision_of = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    for i in range(0, 5, 1):
        vision_of[0][i] = dataset[2 + i]  # of_dx[5]							(float)
    for i in range(0, 5, 1):
        vision_of[1][i] = dataset[7 + i]  # of_dy[5]							(float)
    return(vision_of)

##### ID = 13 ### "vision" ###############################################


def decode_ID13(packet):  # NAVDATA_VISION_TAG
    dataset = struct.unpack_from("HHIiffffifffiIffffffIIff", packet, 0)
    if dataset[1] != 92:
        print("*** ERROR : navdata-vision-Options-Package (ID=13) has the wrong size !!!")
    vision = [0, 0, 0.0, 0.0, 0.0, 0.0, 0, [0.0, 0.0, 0.0], 0,
              0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0, 0, [0.0, 0.0]]
    # vision_state FIXME: What are the meanings of the tags ?
    vision[0] = dataset[2]
    vision[1] = dataset[3]				# vision_misc							(int32)
    vision[2] = dataset[4]				# vision_phi_trim						(float)
    vision[3] = dataset[5]				# vision_phi_ref_prop					(float)
    vision[4] = dataset[6]				# vision_theta_trim						(float)
    vision[5] = dataset[7]				# vision_theta_ref_prop					(float)
    vision[6] = dataset[8]				# new_raw_picture						(int32)
    for i in range(0, 3, 1):
        vision[7][i] = dataset[9 + i]		# theta/phi/psi_capture					(float)
    vision[8] = dataset[12]				# altitude_capture						(int32)
    for i in range(0, 21, 1):						# Calculating milisecond-part
        vision[9] += ((dataset[13] >> i & 1) * (2 ** i))
    vision[9] /= 1000000
    for i in range(21, 32, 1):						# Calculating second-part
        # time_capture			(float)
        vision[9] += (dataset[13] >> i & 1) * (2 ** (i - 21))
    for i in range(0, 3, 1):
        vision[10][i] = dataset[14 + i]  # velocities[xyz]						(float)
    for i in range(0, 3, 1):
        vision[11][i] = dataset[17 + i]  # delta_phi/theta/psi					(float)
    vision[12] = dataset[20]			# gold_defined							(uint32)
    vision[13] = dataset[21]			# gold_reset							(uint32)
    vision[14][0] = dataset[22]			# gold_x								(float)
    vision[14][1] = dataset[23]			# gold_y								(float)
    return(vision)

##### ID = 14 ### "vision_perf" ###############################################


def decode_ID14(packet):  # NAVDATA_VISION_PERF_TAG
    dataset = struct.unpack_from("HHffffffffffffffffffffffffff", packet, 0)
    if dataset[1] != 108:
        print("*** ERROR : navdata-vision_of-Options-Package (ID=14) has the wrong size !!!")
    vision_perf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    vision_perf[0] = dataset[2]				# time_szo								(float)
    vision_perf[1] = dataset[3]				# time_corners							(float)
    vision_perf[2] = dataset[4]				# time_compute							(float)
    vision_perf[3] = dataset[5]				# time_tracking							(float)
    vision_perf[4] = dataset[6]				# time_trans							(float)
    vision_perf[5] = dataset[7]				# time_update							(float)
    for i in range(0, 20, 1):
        vision_perf[6][i] = dataset[8 + i]  # time_custom[20]						(float)
    return(vision_perf)

##### ID = 15 ### "trackers_send" #############################################


def decode_ID15(packet):  # NAVDATA_TRACKERS_SEND_TAG
    dataset = struct.unpack_from(
        "HHiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii", packet, 0)
    if dataset[1] != 364:
        print("*** ERROR : navdata-trackers_send-Options-Package (ID=15) has the wrong size !!!")
    DEFAULT_NB_TRACKERS_WIDTH = 6
    DEFAULT_NB_TRACKERS_HEIGHT = 5
    limit = DEFAULT_NB_TRACKERS_WIDTH * DEFAULT_NB_TRACKERS_HEIGHT
    trackers_send = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [
        0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
    for i in range(0, limit, 1):
        trackers_send[0][i] = dataset[2 + i]  # locked[limit]				(int32)
    for i in range(0, limit, 1):
        # point[x[limit],y[limit]]		(int32)
        trackers_send[1][i][0] = dataset[32 + (i * 2)]
        trackers_send[1][i][1] = dataset[33 + (i * 2)]
    return(trackers_send)

##### ID = 16 ### "vision_detect" #############################################


def decode_ID16(packet):  # NAVDATA_VISION_DETECT_TAG
    dataset = struct.unpack_from(
        "HHIIIIIIIIIIIIIIIIIIIIIIIIIffffIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", packet, offsetND)
    if dataset[1] != 328:
        print("*** ERROR : navdata-vision_detect-Package (ID=16) has the wrong size !!!")
    vision_detect = [0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0, 0, 0, 0]]
    # Max marker detection in one picture: 4
    vision_detect[0] = dataset[2]									 		# nb_detected						(uint32)
    for i in range(0, 4, 1):
        vision_detect[1][i] = dataset[3 + i]			# type[4]							(uint32)
    for i in range(0, 4, 1):
        vision_detect[2][i] = dataset[7 + i]			# xc[4]								(uint32)
    for i in range(0, 4, 1):
        vision_detect[3][i] = dataset[11 + i]			# yc[4]								(uint32)
    for i in range(0, 4, 1):
        vision_detect[4][i] = dataset[15 + i]			# width[4]							(uint32)
    for i in range(0, 4, 1):
        vision_detect[5][i] = dataset[19 + i]			# height[4]							(uint32)
    for i in range(0, 4, 1):
        vision_detect[6][i] = dataset[23 + i]			# dist[4]							(uint32)
    for i in range(0, 4, 1):
        # orientation_angle[4]				(float)
        vision_detect[7][i] = dataset[27 + i]
    for i in range(0, 4, 1):
        for j in range(0, 9, 1):
            # rotation[4]						(float 3x3 matrix (11,12,13,21,...)
            vision_detect[8][i][j] = dataset[31 + i + j]
    for i in range(0, 4, 1):
        for j in range(0, 3, 1):
            # rotation[4]						(float 3 vector)
            vision_detect[9][i][j] = dataset[67 + i + j]
    for i in range(0, 4, 1):
        vision_detect[10][i] = dataset[79 + i]		# camera_source[4]					(uint32)
    return(vision_detect)

##### ID = 17 ### "watchdog" #############################################


def decode_ID17(packet):  # NAVDATA_WATCHDOG_TAG
    dataset = struct.unpack_from("HHI", packet, offsetND)
    if dataset[1] != 8:
        print("*** ERROR : navdata-watchdog-Package (ID=17) has the wrong size !!!")
    watchdog = dataset[2]		# watchdog			Watchdog controll [-]				(uint32)
    return(watchdog)

##### ID = 18 ### "adc_data_frame" #######################################


def decode_ID18(packet):  # NAVDATA_ADC_DATA_FRAME_TAG
    dataset = struct.unpack_from(
        "HHIBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", packet, offsetND)
    if dataset[1] != 40:
        print("*** ERROR : navdata-adc_data_frame-Package (ID=18) has the wrong size !!!")
    adc_data_frame = [0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    adc_data_frame[0] = dataset[2]									# version								(uint32)
    for i in range(0, 32, 1):
        adc_data_frame[1][i] = dataset[3 + i]  # data_frame[32]						(uint8)
    return(adc_data_frame)

##### ID = 19 ### "video_stream" #########################################


def decode_ID19(packet):  # NAVDATA_VIDEO_STREAM_TAG
    dataset = struct.unpack_from("HHBIIIIfIIIiiiiiII", packet, offsetND)
    if dataset[1] != 65:
        print("*** ERROR : navdata-video_stream-Package (ID=19) has the wrong size !!!")
    video_stream = [0, 0, 0, 0, 0, 0.0, 0, 0, 0, [0, 0, 0, 0, 0], 0, 0]
    # quant   		quantizer reference used to encode [1:31]   				(uint8)
    video_stream[0] = dataset[2]
    # frame_size	frame size in bytes   										(uint32)
    video_stream[1] = dataset[3]
    # frame_number	frame index   												(uint32)
    video_stream[2] = dataset[4]
    # atcmd_ref_seq	atmcd ref sequence number   								(uint32)
    video_stream[3] = dataset[5]
    # atcmd_mean_ref_gap      mean time between two consecutive atcmd_ref (ms)
    # (uint32)
    video_stream[4] = dataset[6]
    video_stream[5] = dataset[7]  # atcmd_var_ref_gap															(float)
    # atcmd_ref_quality		estimator of atcmd link quality   					(uint32)
    video_stream[6] = dataset[8]
    # Drone 2.0:
    # out_bitrate			measured out throughput from the video tcp socket	(uint32)
    video_stream[7] = dataset[9]
    # desired_bitrate		last frame size generated by the video encoder		(uint32)
    video_stream[8] = dataset[10]
    for i in range(0, 5, 1):
        # data		misc temporary data				(int32)
        video_stream[9][i] = dataset[11 + i]
    # tcp_queue_level		queue usage											(uint32)
    video_stream[10] = dataset[16]
    # fifo_queue_level		queue usage											(uint32)
    video_stream[11] = dataset[17]
    return(video_stream)

##### ID = 20 ### "games" ################################################


def decode_ID20(packet):  # NAVDATA_GAMES_TAG
    dataset = struct.unpack_from("HHII", packet, offsetND)
    if dataset[1] != 12:
        print("*** ERROR : navdata-games-Package (ID=20) has the wrong size !!!")
    games = [0, 0]
    games[0] = dataset[2]  # double_tap_counter 			   							(uint32)
    games[1] = dataset[3]  # finish_line_counter										(uint32)
    return(games)

##### ID = 21 ### "pressure_raw" #########################################


def decode_ID21(packet):  # NAVDATA_PRESSURE_RAW_TAG
    dataset = struct.unpack_from("HHihii", packet, offsetND)
    if dataset[1] != 18:
        print("*** ERROR : navdata-pressure_raw-Package (ID=21) has the wrong size !!!")
    pressure_raw = [0, 0, 0, 0]
    pressure_raw[0] = dataset[2]  # up 			   									(int32)
    pressure_raw[1] = dataset[3]  # ut												(int16)
    pressure_raw[2] = dataset[4]  # Temperature_meas 									(int32)
    pressure_raw[3] = dataset[5]  # Pression_meas										(int32)
    return(pressure_raw)

##### ID = 22 ### "magneto" ##############################################


def decode_ID22(packet):  # NAVDATA_MAGNETO_TAG
    dataset = struct.unpack_from("HHhhhffffffffffffBifff", packet, offsetND)
    if dataset[1] != 83:
        print("*** ERROR : navdata-magneto-Package (ID=22) has the wrong size !!!")
    magneto = [[0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0]
    for i in range(0, 3, 1):
        magneto[0][i] = dataset[2 + i]  # mx/my/mz											(int16)
    for i in range(0, 3, 1):
        # magneto_raw		magneto in the body frame [mG]	(vector float)
        magneto[1][i] = dataset[5 + i]
    for i in range(0, 3, 1):
        # magneto_rectified									(vector float)
        magneto[2][i] = dataset[8 + i]
    for i in range(0, 3, 1):
        # magneto_offset									(vector float)
        magneto[3][i] = dataset[11 + i]
    magneto[4] = dataset[14]								# heading_unwrapped 								(float)
    magneto[5] = dataset[15]								# heading_gyro_unwrapped							(float)
    magneto[6] = dataset[16]								# heading_fusion_unwrapped 							(float)
    magneto[7] = dataset[17]								# magneto_calibration_ok							(char)
    magneto[8] = dataset[18]								# magneto_state 									(uint32)
    magneto[9] = dataset[19]								# magneto_radius									(float)
    magneto[10] = dataset[20]								# error_mean 										(float)
    magneto[11] = dataset[21]								# error_var											(float)
    return(magneto)

##### ID = 23 ### "wind_speed" ################################################


def decode_ID23(packet):  # NAVDATA_WIND_TAG
    dataset = struct.unpack_from("HHfffffffffffff", packet, offsetND)
    if dataset[1] != 56 and dataset[1] != 64:
        print("*** ERROR : navdata-wind_speed-Package (ID=23) has the wrong size !!!")
    wind_speed = [0.0, 0.0, [0.0, 0.0], [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    wind_speed[0] = dataset[2]							# wind_speed 			   					(float)
    wind_speed[1] = dataset[3]							# wind_angle								(float)
    wind_speed[2][0] = dataset[4]							# wind_compensation_theta 					(float)
    wind_speed[2][1] = dataset[5]							# wind_compensation_phi						(float)
    for i in range(0, 6, 1):
        wind_speed[3][i] = dataset[6 + i]  # state_x[1-6]								(float)
    for i in range(0, 3, 1):
        wind_speed[4][i] = dataset[7 + i]  # magneto_debug[1-3]						(float)
    return(wind_speed)

##### ID = 24 ### "kalman_pressure" ###########################################


def decode_ID24(packet):  # NAVDATA_KALMAN_PRESSURE_TAG
    dataset = struct.unpack_from("HHffffffffff?f?ff??", packet, offsetND)
    if dataset[1] != 72:
        print("*** ERROR : navdata-wind_speed-Package (ID=24) has the wrong size !!!")
    kalman_pressure = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0, 0.0, False, 0.0, 0.0, False, False]
    kalman_pressure[0] = dataset[2]  # offset_pressure 			   					(float)
    kalman_pressure[1] = dataset[3]  # est_z											(float)
    kalman_pressure[2] = dataset[4]  # est_zdot 										(float)
    kalman_pressure[3] = dataset[5]  # est_bias_PWM 									(float)
    kalman_pressure[4] = dataset[6]  # est_biais_pression							(float)
    kalman_pressure[5] = dataset[7]  # offset_US 			   						(float)
    kalman_pressure[6] = dataset[8]  # prediction_US									(float)
    kalman_pressure[7] = dataset[9]  # cov_alt 										(float)
    kalman_pressure[8] = dataset[10]  # cov_PWM										(float)
    kalman_pressure[9] = dataset[11]  # cov_vitesse									(float)
    kalman_pressure[10] = dataset[12]  # bool_effet_sol								(bool)
    kalman_pressure[11] = dataset[13]  # somme_inno									(float)
    kalman_pressure[12] = dataset[14]  # flag_rejet_US									(bool)
    kalman_pressure[13] = dataset[15]  # u_multisinus									(float)
    kalman_pressure[14] = dataset[16]  # gaz_altitude									(float)
    kalman_pressure[15] = dataset[17]  # Flag_multisinus								(bool)
    kalman_pressure[16] = dataset[18]  # Flag_multisinus_debut							(bool)
    return(kalman_pressure)

##### ID = 25 ### "hdvideo_stream" ############################################


def decode_ID25(packet):  # NAVDATA_HDVIDEO-TAG
    dataset = struct.unpack_from("HHfffffff", packet, offsetND)
    if dataset[1] != 32:
        print("*** ERROR : navdata-hdvideo_stream-Package (ID=25) has the wrong size !!!")
    hdvideo_stream = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    hdvideo_stream[0] = dataset[2]  # hdvideo_state 			   					(float)
    hdvideo_stream[1] = dataset[3]  # storage_fifo_nb_packets						(float)
    hdvideo_stream[2] = dataset[4]  # storage_fifo_size 							(float)
    # usbkey_size 			USB key in kb (no key=0)(float)
    hdvideo_stream[3] = dataset[5]
    # usbkey_freespace		USB key in kb (no key=0)(float)
    hdvideo_stream[4] = dataset[6]
    # frame_number                    PaVE field of the frame starting to be
    # encoded for the HD stream (float)
    hdvideo_stream[5] = dataset[7]
    hdvideo_stream[6] = dataset[8]  # usbkey_remaining_time	[sec]					(float)
    return(hdvideo_stream)

##### ID = 26 ### "wifi" ######################################################


def decode_ID26(packet):  # NAVDATA_WIFI_TAG
    dataset = struct.unpack_from("HHI", packet, offsetND)
    if dataset[1] != 8:
        print("*** ERROR : navdata-wifi-Package (ID=26) has the wrong size !!!")
    wifi = dataset[2]  # link_quality 			   								(uint32)
    return(wifi)

##### ID = 27 ### "zimmu_3000" ################################################


def decode_ID27(packet):  # NAVDATA_ZIMU_3000_TAG
    dataset = struct.unpack_from("HHif", packet, offsetND)
    if dataset[1] != 12 and dataset[1] != 216:		# 216 since firmware 2.4.8 ?
        print("*** ERROR : navdata-zimmu_3000-Package (ID=27) has the wrong size !!!")
    zimmu_3000 = [0, 0.0]
    zimmu_3000[0] = dataset[2]  # vzimmuLSB 			   							(int32)
    zimmu_3000[1] = dataset[3]  # vzfind 			   								(float)
    return(zimmu_3000)

##### Footer ### "chksum" #####################################################


# Decode Checksum options-package ID=65535
def decode_Footer(packet, allpacket):
    dataset = struct.unpack_from("HHI", packet, offsetND)
    if dataset[1] != 8:
        print("*** ERROR : Checksum-Options-Package (ID=65535) has the wrong size !!!")
    chksum = [0, False]
    chksum[0] = dataset[2]
    sum, plen = 0, len(allpacket) - 8
    for i in range(0, plen, 1):
        sum += ord(allpacket[i])		# Slows down this Navdata-subprocess massivly
    if sum == chksum[0]:
        chksum[1] = True
    return(chksum)


###############################################################################
# Navdata-Decoding
###############################################################################
def getDroneStatus(packet):
    arState = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    checksum = (0, False)
    length = len(packet)
    # Reading (Header, State, Sequence, Vision)
    dataset = struct.unpack_from("IIII", packet, 0)
    offsetND = struct.calcsize("IIII")

# =-
# Decode Options-Packages ###=-
# =-


def getNavdata(packet, choice):
    navdata = {}
    length = len(packet)
    # Reading (Header, State, Sequence, Vision)
    dataset = struct.unpack_from("IIII", packet, 0)
    navdata["state"] = decode_Header(dataset)
    offsetND = struct.calcsize("IIII")
    # Demo-mode contains normally Option-Packages with ID=0 (_navdata_demo_t), ID=16 (seems empty) and ID=65535 (checksum)
    # Full Mode contains
    while offsetND < length:
        # Reading (Header, Length)
        dataset = struct.unpack_from("HH", packet, offsetND)
        if dataset[0] == 0 and choice[0]:
            navdata["demo"] = decode_ID0(packet[offsetND:])
        if dataset[0] == 1 and choice[1]:
            navdata["time"] = decode_ID1(packet[offsetND:])
        if dataset[0] == 2 and choice[2]:
            navdata["raw_measures"] = decode_ID2(packet[offsetND:])
        if dataset[0] == 3 and choice[3]:
            navdata["phys_measures"] = decode_ID3(packet[offsetND:])
        if dataset[0] == 4 and choice[4]:
            navdata["gyros_offsets"] = decode_ID4(packet[offsetND:])
        if dataset[0] == 5 and choice[5]:
            navdata["euler_angles"] = decode_ID5(packet[offsetND:])
        if dataset[0] == 6 and choice[6]:
            navdata["references"] = decode_ID6(packet[offsetND:])
        if dataset[0] == 7 and choice[7]:
            navdata["trims"] = decode_ID7(packet[offsetND:])
        if dataset[0] == 8 and choice[8]:
            navdata["rc_references"] = decode_ID8(packet[offsetND:])
        if dataset[0] == 9 and choice[9]:
            navdata["pwm"] = decode_ID9(packet[offsetND:])
        if dataset[0] == 10 and choice[10]:
            navdata["altitude"] = decode_ID10(packet[offsetND:])
        if dataset[0] == 11 and choice[11]:
            navdata["vision_raw"] = decode_ID11(packet[offsetND:])
        if dataset[0] == 12 and choice[12]:
            navdata["vision_of"] = decode_ID12(packet[offsetND:])
        if dataset[0] == 13 and choice[13]:
            navdata["vision"] = decode_ID13(packet[offsetND:])
        if dataset[0] == 14 and choice[14]:
            navdata["vision_perf"] = decode_ID14(packet[offsetND:])
        if dataset[0] == 15 and choice[15]:
            navdata["trackers_send"] = decode_ID15(packet[offsetND:])
        if dataset[0] == 16 and choice[16]:
            navdata["vision_detect"] = decode_ID16(packet[offsetND:])
        if dataset[0] == 17 and choice[17]:
            navdata["watchdog"] = decode_ID17(packet[offsetND:])
        if dataset[0] == 18 and choice[18]:
            navdata["adc_data_frame"] = decode_ID18(packet[offsetND:])
        if dataset[0] == 19 and choice[19]:
            navdata["video_stream"] = decode_ID19(packet[offsetND:])
        if dataset[0] == 20 and choice[20]:
            navdata["games"] = decode_ID20(packet[offsetND:])
        if dataset[0] == 21 and choice[21]:
            navdata["pressure_raw"] = decode_ID21(packet[offsetND:])
        if dataset[0] == 22 and choice[22]:
            navdata["magneto"] = decode_ID22(packet[offsetND:])
        if dataset[0] == 23 and choice[23]:
            navdata["wind_speed"] = decode_ID23(packet[offsetND:])
        if dataset[0] == 24 and choice[24]:
            navdata["kalman_pressure"] = decode_ID24(packet[offsetND:])
        if dataset[0] == 25 and choice[25]:
            navdata["hdvideo_stream"] = decode_ID25(packet[offsetND:])
        if dataset[0] == 26 and choice[26]:
            navdata["wifi"] = decode_ID26(packet[offsetND:])
        if dataset[0] == 27 and choice[27]:
            navdata["zimmu_3000"] = decode_ID27(packet[offsetND:])
        if dataset[0] == 65535 and choice[28]:
            navdata["chksum"] = decode_Footer(packet[offsetND:], packet)
        offsetND += dataset[1]
    return(navdata)

# =-
# Threads					###=-
# =-


def reconnect(navdata_pipe, commitsuicideND, DroneIP, NavDataPort):
    if not commitsuicideND:
        navdata_pipe.sendto("\x01\x00\x00\x00", (DroneIP, NavDataPort))


def watchdogND(parentPID):
    global commitsuicideND
    while not commitsuicideND:
        time.sleep(1)
        try:
            os.getpgid(parentPID)
        except:
            commitsuicideND = True
# It seems that you just have to reinitialize the network-connection once
# and the drone keeps on sending forever then.


def mainloopND(DroneIP, NavDataPort, parent_pipe, parentPID):
    global commitsuicideND
    something2send, MinimalPacketLength, timetag = False, 30, 0
    packetlist = ["demo", "time", "raw_measures", "phys_measures", "gyros_offsets", "euler_angles", "references", "trims", "rc_references", "pwm", "altitude", "vision_raw", "vision_of", "vision", "vision_perf",
                  "trackers_send", "vision_detect", "watchdog", "adc_data_frame", "video_stream", "games", "pressure_raw", "magneto", "wind_speed", "kalman_pressure", "hdvideo_stream", "wifi", "zimmu_3000", "chksum", "state"]
    choice = [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]
    # This and oneTimeFailOver is necessary because of a bug (?) of AR.Drone
    # sending NavData in DemoMode...
    overallchoice = False
    # ...while setting a configuration the drone sends the next DemoMode-package with just its status.
    oneTimeFailOver = True
    debug = False
    showCommands = False

    # Checks if the main-process is running and sends ITS own PID back
    ThreadWatchdogND = threading.Thread(target=watchdogND, args=[parentPID])
    ThreadWatchdogND.start()

    # Prepare communication-pipes
    pipes = []
    pipes.append(parent_pipe)
    navdata_pipe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    navdata_pipe.setblocking(0)
    navdata_pipe.bind(('', NavDataPort))
    pipes.append(navdata_pipe)

    # start connection
    reconnect(navdata_pipe, commitsuicideND, DroneIP, NavDataPort)
    # Inits the first Network-Heartbeat (2 secs after disconnection the drone
    # stops sending)
    netHeartbeat = threading.Timer(
        2.0, reconnect, [navdata_pipe, commitsuicideND, DroneIP, NavDataPort, ])
    netHeartbeat.start()

    if choice.count(True) > 0:
        overallchoice = True

    while not commitsuicideND:
        # When something is in a pipe...
        in_pipe, out_pipe, dummy2 = select.select(pipes, [], [], 0.5)
        for ip in in_pipe:
            if ip == parent_pipe:
                cmd = parent_pipe.recv()
                if showCommands:
                    print("** Com -> Nav : ", cmd)
                # Signal to stop this process and all its threads
                if cmd == "die!":
                    commitsuicideND = True
                # Enables/disables Debug-bit
                elif cmd == "debug":
                    debug = True
                    print("NavData-Process :    running")
                elif cmd == "undebug":
                    debug = False
                # Enables/disables Debug-bit
                elif cmd == "showCommands":
                    showCommands = True
                elif cmd == "hideCommands":
                    showCommands = False
                elif cmd == "reconnect":
                    reconnect(
                        navdata_pipe, commitsuicideND, DroneIP, NavDataPort)
                # Sets explicitly the value-packages which shall be decoded
                elif cmd[0] == "send":
                    if cmd[1].count("all"):
                        for i in range(0, len(choice), 1):
                            choice[i] = True
                    else:
                        for i in range(0, len(packetlist), 1):
                            if cmd[1].count(packetlist[i]):
                                choice[i] = True
                            else:
                                choice[i] = False
                    if choice.count(True) > 0:
                        overallchoice = True
                    else:
                        overallchoice = False
                # Adds value-packages to the other which shall be decoded
                elif cmd[0] == "add":
                    for i in range(0, len(packetlist), 1):
                        if cmd[1].count(packetlist[i]):
                            choice[i] = True
                    if cmd[1].count("all"):
                        for i in range(0, len(choice), 1):
                            choice[i] = True
                    if choice.count(True) > 0:
                        overallchoice = True
                    else:
                        overallchoice = False
                # Deletes packages from the value-package-list which shall not
                # be decoded anymore
                elif cmd[0] == "block":
                    if cmd[1].count("all"):
                        for i in range(0, len(packetlist), 1):
                            choice[i] = False
                    else:
                        for i in range(0, len(packetlist), 1):
                            if cmd.count(packetlist[i]):
                                choice[i] = False
                    if choice.count(True) > 0:
                        overallchoice = True
                    else:
                        overallchoice = False
            if ip == navdata_pipe:
                try:
                    # Connection is alive, Network-Heartbeat not necessary for
                    # a moment
                    netHeartbeat.cancel()
                    # Receiving raw NavData-Package
                    Packet = navdata_pipe.recv(65535)
                    netHeartbeat = threading.Timer(
                        2.1, reconnect, [navdata_pipe, commitsuicideND, DroneIP, NavDataPort])
                    # Network-Heartbeat is set here, because the drone keeps on
                    # sending NavData (vid, etc you have to switch on)
                    netHeartbeat.start()
                    # Setting up decoding-time calculation
                    timestamp = timetag
                    timetag = time.time()
                    if overallchoice:
                        try:
                            lastdecodedNavData = decodedNavData
                        except:
                            lastdecodedNavData = {}
                    decodedNavData = getNavdata(Packet, choice)
                    state = decodedNavData["state"]
                    # If there is an abnormal small NavPacket, the last
                    # NavPacket will be sent out with an error-tag
                    NoNavData = False
                    if len(Packet) < MinimalPacketLength and overallchoice:
                        decodedNavData, NoNavData = lastdecodedNavData, True
                    dectime = time.time() - timetag
                    # Sends all the data to the mainprocess
                    parent_pipe.send(
                        (decodedNavData, state[0:32], state[32], timestamp, dectime, NoNavData))
                except IOError:
                    pass
    suicideND = True
    netHeartbeat.cancel()
    if debug:
        print("NavData-Process :    committed suicide")


##########################################################################
###### Playground																			######
##########################################################################
if __name__ == "__main__":
    ###
    # Here you can write your first test-codes and play around with them
    ###
    
    try:
        import cv2
        import time, sys
        import ps_drone

        global vidPipePath
        
        drone = ps_drone.Drone()		# Start using drone
        drone.startup()				# Connects to drone and starts subprocesses
        drone.reset()		        	# Always good, at start

        while drone.getBattery()[0] == -1:
            time.sleep(0.1)		# Waits until the drone has done its reset
        time.sleep(0.5)			# Give it some time to fully awake

        # Gives a battery-status
        print("Battery: " + str(drone.getBattery()[0]) + "%  " + str(drone.getBattery()[1]))

        stop = False
        

        drone.setConfigAllID()                                       # Go to multiconfiguration-mode
        drone.sdVideo()                                              # Choose lower resolution (hdVideo() for...well, guess it)
        drone.frontCam()                                             # Choose front view
        CDC = drone.ConfigDataCount
        while CDC == drone.ConfigDataCount:       time.sleep(0.0001) # Wait until it is done (after resync is done)
        drone.startVideo()                                           # Start video-function
        drone.showVideo()

        #        print ("2VidPipePath " + vidPipePath)

        
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
 
        #       print ("Use <space> to toggle front- and groundcamera, any other key to stop")
        IMC =    drone.VideoImageCount                               # Number of encoded videoframes
        stop =   False
        ground = False

        drone_camera_id = "/tmp/dronevid-5817350912-471575.h264"
        print("Drone.vidTemp; " +str(Drone.vidTemp))
        #capture = cv2.VideoCapture(Drone.vidTemp)
        #capture = cv2.VideoCapture(vidPipePath)
        
        # pygame  init
        pygame.init()
        joys = pygame.joystick.Joystick(0)
        pygame.joystick.init()
        joys.init()

        while not stop:


            for e in pygame.event.get():
                if e.type == pygame.locals.JOYHATMOTION:
                    hat = joys.get_hat(0)
                    print(hat)
                    if hat[0] == -1:
                        drone.moveLeft()
                    elif hat[0] == 1:
                        drone.moveRight()
                    elif hat[1] == -1:
                        drone.moveBackward()
                    elif hat[1] == 1:
                        drone.moveForward()
                elif e.type == pygame.locals.JOYBUTTONDOWN:
                    JOYBUTTON_A = 1
                    JOYBUTTON_B = 2
                    JOYBUTTON_Y = 3
                    JOYBUTTON_X = 0
                    JOYBUTTON_LB = 4
                    JOYBUTTON_RB = 5
                    print(str(e.button))
                    if joys.get_button(4):
                        drone.turnLeft()
                    elif joys.get_button(5):
                        drone.turnRight()
                    elif joys.get_button(3):
                        drone.moveUp()
                    elif joys.get_button(1):
                        drone.moveDown()
                    elif joys.get_button(2):
                        drone.hover()

            key = drone.getKey()                                     # Gets a pressed key            
            if key == " ":
                if drone.NavData["demo"][0][2] and not drone.NavData["demo"][0][3]:
                    drone.takeoff()
                else:
                    drone.land()
            elif key == "0":
                drone.hover()
            elif key == "w":
                drone.moveForward()
            elif key == "s":
                drone.moveBackward()
            elif key == "a":
                drone.moveLeft()
            elif key == "d":
                drone.moveRight()
            elif key == "q":
                drone.turnLeft()
            elif key == "e":
                drone.turnRight()
            elif key == "7":
                drone.turnAngle(-10, 1)
            elif key == "9":
                drone.turnAngle(10, 1)
            elif key == "4":
                drone.turnAngle(-45, 1)
            elif key == "6":
                drone.turnAngle(45, 1)
            elif key == "1":
                drone.turnAngle(-90, 1)
            elif key == "3":
                drone.turnAngle(90, 1)
            elif key == "8":
                drone.moveUp()
            elif key == "2":
                drone.moveDown()
            elif key == "*":
                drone.doggyHop()
            elif key == "+":
                drone.doggyNod()
            elif key == "-":
                drone.doggyWag()
            elif key != "":
                stop = True

        # Gives a battery-status
        print("Batterie: " + str(drone.getBattery()[0]) + "%  " + str(drone.getBattery()[1]))
    except pygame.error:
        print("Cannot find any joysticks")
        sys.exit()


DRONE = None
def drone_factory():
    global DRONE
    if not DRONE:
        DRONE = Drone()
        DRONE.startup()
    return DRONE


class flying():
    def __init__(self):
        self.d = drone_factory()

    def __enter__(self):
        self.d.reset()
        while self.d.NavData['demo'][0][2] == 0:
            time.sleep(.1)  # wait for reset
        print("BAT", self.d.getBattery())
        self.d.takeoff()
        time.sleep(7)  # give time to take off
        return self.d

    def __exit__(self, type, value, tb):
        self.d.land()
        self.d.reset()

def mapdelay(func, seq, delay=.1):
    for item in seq:
        yield func(seq)
        time.sleep(delay)
