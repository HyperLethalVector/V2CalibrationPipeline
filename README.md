This is a fork from the North Star main branch: https://github.com/BryanChrisBrown/ProjectNorthStar/tree/realsense-integration

# Headset 2D Calibration Routine

This is a guide for the new, simplified Calibration routine.  Rather than align the physical components within a North Star/Ariel headset to minimize the geometric raytracing error, this calibration instead grafts a camera calibration onto the headset itself.

Before we start, you will need to ensure that you have a:

 - A Project North Star/ Ariel Headset
 - A Project North Star/ Ariel Calibration Stand
 - A Calibration Checkerboard (can be displayed on a monitor)

If you're using the t261, you can skip the sensor calibration~

Note that this new calibration routine does not require an external monitor.
## Calibrating the Sensors

 - **captureChessboards.py** 
  - Running this script displays images from your stereo camera.  Show your chessboard to the camera and press **Z** to capture it from various angles.  Get ~30 shots.
  - These images are stored in **./chessboardImages/** as .pngs

 - **viewChessboards.py** 
  - Running this script will simply display the captured images and chessboard patterns on top of each one.  Use this to find misbehaving images where the where the chessboard grid is misaligned between the left and right images.

 - **calibrateChessboards.py** 
  - Running this script will load in your captured chessboard images and calibrate your stereo camera using OpenCV's Fisheye Calibration.


## Calibrating the Headset
 - **captureGraycodes.py**
  - Ensure that your headset is placed on the calibration stand, with the stand's camera looking through it where the users' eyes will be.
  - Additionally ensure that your headset is plugged in and displaying imagery from your desktop.
  - It helps to place a piece of cloth over the rig to shield the cameras + headset from ambient light.
  - Running this script will display a sequence of graycodes on your North Star, capturing them at the same time.
  - The sequence of binary codes will culminate in a 0-1 UV mapping, saved to **"./WidthCalibration.png"** **"./HeightCalibration.png"** in your main folder.
  - This script has now been modified to do each eye separately (for optics that have bleed between the lenses)
  - this script will run the **calibrateGraycodes.py** script for you, no need to worry, just open the **"OpticalCalibrations.json"** file in your main folder.


When you are finished, you may paste the output of the calibrateGraycodes.py into [this diagnostic shadertoy](https://www.shadertoy.com/view/wsscD4) to check for alignment.

Additionally, there should be a `NorthStarCalibration.json` in this directory which you may use in the Unity implementation.
