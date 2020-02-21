# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import time
import logging
logging.basicConfig(level=logging.INFO)

def encode_face_func(dataset,num_jitters,detection_method,image_acceptable_width=2560):
    args["dataset"]=dataset
    args["num_jitters"]=num_jitters
    args["detection_method"]=detection_method


    # grab the paths to the input images in our dataset
    logging.info("quantifying faces...")

    tstart=time.time()
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    image_acceptable_width=image_acceptable_width

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        logging.info("processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        hight,width=image.shape[:2]
        if image.shape[1]>image_acceptable_width:
            factor = image_acceptable_width/image.shape[1]
            width = image_acceptable_width
            hight = round(hight*factor)
        image = cv2.resize(image,(width, hight), interpolation = cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model=args["detection_method"])

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes,num_jitters=args["num_jitters"])

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    logging.info("serializing encodings...")
    dataset_location=args["dataset"].replace('/','')
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(f'encoding_{args["detection_method"]}_nj{args["num_jitters"]}_{dataset_location}.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()
    tend=time.time()
    logging.info(f'Total encoing time : {(tend-tstart)}')

if __name__=="__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
    ap.add_argument("-nj", "--num-jitters", type=int, default=1,
        help="input the number of num_jitters")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-iw", "--image_width",type=int,default=2560,
        help="input the acceptable width for your device")
    args = vars(ap.parse_args())
    encode_face_func(args["dataset"],args["num_jitters"],args["detection_method"],args["image_width"])
