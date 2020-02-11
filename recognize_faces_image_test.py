# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import logging
import time
import pandas as pd
import os
from collections import Counter


logging.basicConfig(level=logging.INFO)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


tstart=time.time()
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# load the input image and convert it from BGR to RGB
logging.debug(f'read image path:{args["image"]}\n')
image = cv2.imread(args["image"])
hight,width=image.shape[:2]
logging.debug(f'image ori width:{width},image ori hight:{hight}\n')
if image.shape[1]>1024:
    factor = 1024/image.shape[1]
    logging.debug(f'resize fector:{factor}\n')
    width = 1024
    hight = round(hight*factor)
logging.debug(f'image transfer width:{width},image transfer hight:{hight}\n')
image = cv2.resize(image,(width, hight), interpolation = cv2.INTER_CUBIC)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
    model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
    # attempt to match each face in the input image to our known
    # encodings
    matches = face_recognition.compare_faces(data["encodings"],
        encoding,tolerance=0.4)
    logging.debug(f'mathces : {matches}\n')
    name = "unknown"

    # check to see if we have found a match
    if True in matches:
        # find the indexes of all matched faces then initialize a
        # dictionary to count the total number of times each face
        # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        logging.debug(f'matchedIdxs: {matchedIdxs}\n')
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        logging.debug(f'counts : {counts}\n')

        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)
    
    # update the list of names
    names.append(name)

imagepathTail=os.path.split(args["image"])[1]
count = Counter(names)
label = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/testSetTable.csv')
label = label.loc[label['filename']==imagepathTail,:]
TP,FP,TN,FN=[0,0,0,0]

for key in label.columns[2:6]:
    print(count.get(key),label[key].values.item(0), key,'\n============================================\n')
    if key=='unknown':
        if count.get(key) and count.get(key)>=label[key].values.item(0):
            TN =label[key].values.item(0)
            FN = count.get(key)-label[key].values.item(0)
        elif count.get(key) and count.get(key)<label[key].values.item(0):
            TN = count.get(key)
            FN = 0
    elif key!='unknown':
        if count.get(key) and count.get(key)>=label[key].values.item(0):
            TP = label[key].values.item(0)
            FP = count.get(key)-label[key].values.item(0)
        elif count.get(key) and count.get(key)<label[key].values.item(0):
            TP = count.get(key)
            FP = 0

    # if count.get(key) and key!='unknown' and result==0:
    #     TP = result+TP
    #     print('tp+1',count.get(key),label[key].values.item(0), key)
    # elif count.get(key) and key!='unknown' and count.get(key)!=label[key].values.item(0):
    #     FP+=1
    #     print('fp+1',count.get(key),label[key].values.item(0), key)
    # elif count.get(key) and key=='unknown' and count.get(key)==label[key].values.item(0):
    #     TN+=1
    #     print('tn+1',count.get(key),label[key].values.item(0), key)
    # elif count.get(key) and key=='unknown' and count.get(key)!=label[key].values.item(0):
    #     FN+=1
    #     print('fn+1',count.get(key),label[key].values.item(0), key)
        
logging.info(f'True Postive : {TP} , False Postive : {FP} , True Negtive : {TN} , False Negtive {FN}')

logging.debug(f'workingDir : {os.getcwd()} , filename : {__file__} , dirname : {os.path.dirname(__file__)} , abspath : {os.path.abspath(__file__)} , base : {os.path.basename(__file__)} , dir(abs) : {os.path.dirname(os.path.abspath(__file__))}\n')
logging.debug(f'file : {imagepathTail}\n')
logging.info(f'pridct names:{names},Count : {count}\n')

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2)
tend=time.time()

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
logging.info(f'recognize itme : {tend-tstart}\n')