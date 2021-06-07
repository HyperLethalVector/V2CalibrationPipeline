import os
import numpy as np
import cv2
import math
import time
import pyrealsense2 as rs2
import intelutils

rigel = False
realsense = False
mock = True

if rigel:
  frameWidth  = 800
  frameHeight = 800
elif realsense:
  frameWidth = 848
  frameHeight = 800
elif mock:
  frameWidth = 512
  frameHeight = 512

northStarSize = (2880, 1600)
if mock:
  northStarSize = (1024, 512)



cv2.namedWindow      ("Graycode Viewport", 0)#cv2.WINDOW_NORMAL)
cv2.moveWindow       ("Graycode Viewport", 1920, 0)
cv2.setWindowProperty("Graycode Viewport", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
darkFrameBuffer    = np.zeros((720, 1280), dtype=np.uint8)

#Initialize the Stereo Camera's feed
if rigel:
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH , frameWidth)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameWidth)
  cap.set(cv2.CAP_PROP_CONVERT_RGB , False)
elif realsense:
  print("creating intel thread (at %f)" % time.time())
  cap = intelutils.intelCamThread(frame_callback = lambda frame: None)
  print("starting intel thread (at %f)" % time.time())
  cap.start()
  print("started intel thread (at %f)" % time.time())
elif mock:
  class MockCapture:
    def read(self):
      return True, displayedBuffer
  cap = MockCapture()

if rigel:
  # Turn the Rigel Exposure Up
  os.system(".\LFTool.exe xu set leap 30 " + str(6111) + "L")
# if realsense:



# Initialize 3D Visualizer
#should do a for loop, one for each eye
key = cv2.waitKey(1)
for leftEye in range(2):
  whiteBrightness = 127
  halfpoint = int(northStarSize[0]/2)
  fullpoint = int(northStarSize[0])
  allWhite           = np.ones       ((northStarSize[1], northStarSize[0]), dtype=np.uint8) * 100
  continuum          = np.arange     (0, 256,         dtype=np.uint8)
  continuum          = np.bitwise_xor(continuum, continuum//2) # Binary to Gray
  widthContinuum     = np.zeros      (allWhite.shape, dtype=np.uint8)
  widthContinuum[:, : int(northStarSize[0] / 2)]   = cv2.resize(continuum[None, :], (int(northStarSize[0] / 2), northStarSize[1]), interpolation=cv2.INTER_NEAREST)
  widthContinuum[:,   int(northStarSize[0] / 2) :] = widthContinuum[:, : int(northStarSize[0] / 2)]
  heightContinuum    = cv2.resize   (continuum      [:, None      ], northStarSize, interpolation=cv2.INTER_NEAREST)
  widthBits          = np.unpackbits(widthContinuum [:,    :, None].astype(np.uint8), axis=-1)
  heightBits         = np.unpackbits(heightContinuum[:,    :, None].astype(np.uint8), axis=-1)
  widthMeasuredBits  = np.zeros ((frameHeight, frameWidth * 2, 8), dtype=np.uint8)
  heightMeasuredBits = np.zeros ((frameHeight, frameWidth * 2, 8), dtype=np.uint8)
  lastResult         = np.zeros ((frameHeight, frameWidth * 2))
  displayedBuffer    = 100 - allWhite
  frameCount = -3
  captureNum = 0
  bitField   = 0
  stage      = 0
  while (not (key & 0xFF == ord('q'))):
      key = cv2.waitKey(1)
#      print("getting a frame")
      
      # Capture frame-by-frame
      newFrame, frame = cap.read()
#      print("got a frame at %f" % time.time())
      if (newFrame):
          time.sleep(0.1)
 #         print("got a new frame")
          # Reshape our one-dimensional image into a two-channel side-by-side view of the Rigel's feed
          frame       = np.reshape  (frame, (frameHeight, frameWidth * 2))
          frame_color = cv2.cvtColor(frame , cv2.COLOR_GRAY2BGR)
          bitIndex = int((stage-1)/2)
          if frameCount%6 is 0 and frameCount > 0: #key & 0xFF == ord('z'): #
            if stage is 0:
              # Capture all Black Buffer
              darkFrameBuffer = frame.copy()
              # Set Display to White
              displayedBuffer = allWhite
              if leftEye == 0:
                displayedBuffer[:,0:halfpoint] = 0
              else:
                displayedBuffer[:,halfpoint:fullpoint] = 0              
            elif stage is 1:
              # Calculate the Monitor Mask and display it
              mask = cv2.threshold(cv2.subtract(frame, darkFrameBuffer), thresh=53, maxval=1, type=cv2.THRESH_BINARY)[1]
              if leftEye == 0:
                mask[:,0:halfpoint] = 0
                cv2.imwrite("maskLeft.png",mask*whiteBrightness)
              else:
                mask[:,halfpoint:fullpoint] = 0                             
                cv2.imwrite("maskRight.png",mask*whiteBrightness)
              #cv2.imshow("Graycode Display", mask * whiteBrightness)
              
              # Begin displaying the Width Bits
              displayedBuffer = widthBits [:, :, bitIndex] * whiteBrightness
              if leftEye == 0:
                displayedBuffer[:,0:halfpoint] = 0
              else:
                displayedBuffer[:,halfpoint:fullpoint] = 0              

            elif stage < 17:
              if stage % 2 is 0:
                darkFrameBuffer = frame.copy()
                displayedBuffer = (1 - widthBits [:, :, bitIndex]) * whiteBrightness
                if leftEye == 0:
                  displayedBuffer[:,0:halfpoint] = 0
                else:
                  displayedBuffer[:,halfpoint:fullpoint] = 0              
              else:
                bitmask = cv2.threshold(cv2.subtract(frame, darkFrameBuffer), thresh=1, maxval=1, type=cv2.THRESH_BINARY)[1]
                #cv2.imshow("Graycode Display", bitmask.copy() * mask * whiteBrightness)

                # Add this bitmask to the built up bitmask
                lastResult = bitmask == lastResult # xor with last bitmask - Grey -> binary
                widthMeasuredBits[:, :, bitIndex-1] = lastResult

                displayedBuffer =      widthBits [:, :, bitIndex]  * whiteBrightness
                if leftEye == 0:
                  displayedBuffer[:,0:halfpoint] = 0
                else:
                  displayedBuffer[:,halfpoint:fullpoint] = 0              
            elif stage < 33:
              if stage is 17:
                # The Width Bits have finished displaying, we can now pack the graycode bits back into a byte mapping
                if leftEye == 0:
                  calculatedWidthContinuum = np.packbits(widthMeasuredBits, axis=-1)[:, :, 0] * mask
                else:
                  calculatedWidthContinuum = np.packbits(widthMeasuredBits, axis=-1)[:, :, 0] * mask                    
                # This is successful!
                #cv2.imshow("Graycode Width Continuum", cv2.applyColorMap(calculatedWidthContinuum, cv2.COLORMAP_JET))
                if leftEye == 0:
                  cv2.imwrite("./WidthCalibration-Left.png", calculatedWidthContinuum[:,:])                  
                else:
                  cv2.imwrite("./WidthCalibration-Right.png", calculatedWidthContinuum[:,:])

                # reset everything - easiest to do here
                lastResult.fill(0)
                darkFrameBuffer.fill(0)
                frame.fill(0)

              if stage % 2 is 0:
                darkFrameBuffer = frame.copy()
                displayedBuffer = (1 - heightBits [:, :, bitIndex-8]) * whiteBrightness
                if leftEye == 0:
                  displayedBuffer[:,0:halfpoint] = 0
                else:
                  displayedBuffer[:,halfpoint:fullpoint] = 0              
              else:
                bitmask = cv2.threshold(cv2.subtract(frame, darkFrameBuffer), thresh=1, maxval=1, type=cv2.THRESH_BINARY)[1]
                #cv2.imshow("Graycode Display", bitmask.copy() * mask * whiteBrightness)

                lastResult = bitmask == lastResult # xor with last bitmask - Grey -> binary
                heightMeasuredBits[:, :, bitIndex-9] = lastResult

                displayedBuffer =      heightBits [:, :, bitIndex-8]  * whiteBrightness
                if leftEye == 0:
                  displayedBuffer[:,0:halfpoint] = 0
                else:
                  displayedBuffer[:,halfpoint:fullpoint] = 0              
            else:
                # The Width Bits have finished displaying, we can now pack the graycode bits back into a byte mapping
                if leftEye == 0:
                  calculatedHeightContinuum = np.packbits(heightMeasuredBits, axis=-1)[:, :, 0] * mask
                else:
                  calculatedHeightContinuum = np.packbits(heightMeasuredBits, axis=-1)[:, :, 0] * mask                  
                # This is successful!
                #cv2.imshow("Graycode Height Continuum", cv2.applyColorMap(calculatedHeightContinuum, cv2.COLORMAP_JET))

                if leftEye == 0:
                  cv2.imwrite("./HeightCalibration-Left.png", calculatedHeightContinuum[:,:])
                  break
                else:
                  cv2.imwrite("./HeightCalibration-Right.png", calculatedHeightContinuum[:,:])                  
                  print("Finished, should be merging the two now")
                  imgHeightLeft = cv2.imread("./HeightCalibration-Left.png",0)
                  imgHeightRight = cv2.imread("./HeightCalibration-Right.png",0)
                  imgWidthLeft = cv2.imread("./WidthCalibration-Left.png",0)
                  imgWidthRight = cv2.imread("./WidthCalibration-Right.png",0)
 #                 h , w = imgHeightLeft.shape
#                  wHalf = int(w/2)
#                  cmbImgHeight = calculatedHeightContinuum
#                  cmbImgHeight[0:h,:wHalf] = imgHeightLeft[0:h,:wHalf]
#                  cmbImgHeight[0:h,wHalf:w] = imgHeightRight[0:h,wHalf:w]

#                  cmbImgWidth = calculatedWidthContinuum
#                  cmbImgWidth[0:h,:wHalf] = imgWidthLeft[0:h,:wHalf]
#                  cmbImgWidth[0:h,wHalf:w] = imgWidthRight[0:h,wHalf:w]
                  cv2.imwrite("./HeightCalibration.png",cv2.addWeighted(imgHeightLeft,1.0,imgHeightRight,1.0,0))
                  cv2.imwrite("./WidthCalibration.png",cv2.addWeighted(imgWidthLeft,1.0,imgWidthRight,1.0,0))
                  cv2.destroyAllWindows()
                  exit()                 
                
              # Display the height bits
            stage += 1

          cv2.imshow("Graycode Viewport", displayedBuffer)
          
          # Display the resulting frame
          cv2.imshow('Frame', frame_color)#cv2.resize(frame_color,  (384*2,384)))
          if rigel and (frameCount is 1):
            # Turn the Rigel LED Off
            os.system(".\LFTool.exe xu set leap 27 0")

          frameCount = frameCount + 1

cv2.destroyAllWindows()