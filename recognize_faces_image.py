# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import logging
import time
from collections import Counter

logging.basicConfig(level=logging.INFO,format='%(asctime)s--%(module)s--%(levelname)s\n%(message)s')
args={}
def readPara(encodings,image,detection_method,tolerance):
    args["encodings"]=encodings
    args["image"]=image
    args["detection_method"]=detection_method
    args["Tolerance"]=tolerance
    
    logging.info(f'{args}')
    # load the known faces and embeddings
    logging.info("loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())

    # load the input image and convert it from BGR to RGB
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
    logging.info("recognizing faces...")
    tdetectionStart=time.time()
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    tdetectionEnd=time.time()
    tEncodingStart=time.time()
    encodings = face_recognition.face_encodings(rgb, boxes)
    tEncodingEnd=time.time()
    
    tCompareStart=time.time()
    # initialize the list of names for each face detected
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        
        matches = face_recognition.compare_faces(data["encodings"],
            encoding,tolerance=args["Tolerance"])
        name = "unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        names.append(name)
    tCompareEnd=time.time()
    
    count = Counter(names)
    logging.debug(f'Count : {count}\n')

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    cv2.imwrite('Image.jpg',image)
    td=tdetectionEnd-tdetectionStart
    te=tEncodingEnd-tEncodingStart
    tc=tCompareEnd-tCompareStart
    tt=td+te+tc
    logging.info(f'\nDetection time : {td} \nEncoding time : {te} \nComapare time : {tc} \nTotal recognize time : {tt}\n')
    return count


