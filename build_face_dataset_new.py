# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml 
#                              --output dataset/yourname

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import face_recognition

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = 'cnn',
    help = "Face detect model")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory")
args = vars(ap.parse_args())


# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, clone it, (just
    # in case we want to write it to disk), and then resize the frame
    # so we can apply face detection faster
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    # Find all the faces and face encodings in the image
    face_locations = face_recognition.face_locations(frame,model=args['model'])
    print("Found {} faces in image.".format(len(face_locations)))
    
    # Loop over each face found in the frame to see if it's someone we know.
    for face in face_locations:
        (top, right, bottom, left) = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(
            str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
        print(f'{total}')
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
