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


def resize(image,image_acceptable_width):
    hight,width=image.shape[:2]
    logging.debug(f'image ori width:{width},image ori hight:{hight}\n')
    if image.shape[1]>image_acceptable_width:
        factor = image_acceptable_width/image.shape[1]
        logging.debug(f'resize fector:{factor}\n')
        width = image_acceptable_width
        hight = round(hight*factor)
    logging.debug(f'image transfer width:{width},image transfer hight:{hight}\n')
    image = cv2.resize(image,(width, hight), interpolation = cv2.INTER_CUBIC)
    return image

def show_image(image,boxes):
    for (top, right, bottom, left) in boxes:
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)
        y = top - 15 if top - 15 > 15 else top + 15
    image=resize(image,800)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def revise_csv(DataFrame,file_name,reviseArgs):
    logging.info('input the number of hao in this picture : ')
    reviseArgs['hao']=int(input())
    logging.info('input the number of ywt in this picture : ')
    reviseArgs['ywt']=int(input())
    logging.info('input the number of ford in this picture : ')
    reviseArgs['ford']=int(input())
    logging.info('input the number of unknown people in this picture : ')
    reviseArgs['unknown']=int(input())
    DataFrame.loc[label['filename']==file_name,'hao']=reviseArgs['hao']
    DataFrame.loc[label['filename']==file_name,'ywt']=reviseArgs['ywt']
    DataFrame.loc[label['filename']==file_name,'ford']=reviseArgs['ford']
    DataFrame.loc[label['filename']==file_name,'unknown']=reviseArgs['unknown']
    DataFrame.to_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/testSetTable.csv',index=True,header=True)
    return reviseArgs['hao'],reviseArgs['ywt'],reviseArgs['ford'],reviseArgs['ford']

def rm_img_ow_csv(DataFrame,file_name):
    cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/photo_cant_read/'+file_name.split('.')[0]+"_pridict.jpg",image)
    DataFrame=DataFrame.drop(index=labelExt.index.values.item())
    logging.debug(f'label after drop : \n{label}')
    DataFrame.to_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/testSetTable.csv',index=True,header=True)
    os.replace(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/photo/'+file_name,os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/photo_cant_read/'+file_name)

try:
    logging.basicConfig(level=logging.DEBUG)


    logging.debug(f'\nworkingDir : {os.getcwd()} \n filename : {__file__} \n dirname : {os.path.dirname(__file__)} \n abspath : {os.path.abspath(__file__)} \n base : {os.path.basename(__file__)} \n dir(abs) : {os.path.dirname(os.path.abspath(__file__))}\n')


    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", nargs='*', type=str, required=True,
        help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-r", "--resolution", type=str, required=True,
        help="input the resolution")
    # ap.add_argument("-t", "--tolerance", type=str, required=True,
    #     help="input the tolerance")
    args = vars(ap.parse_args())

    input_image_list=args['image'][0].split(',')
    image_acceptable_width=int(args['resolution'])
    TP_sum,FP_sum,TN_sum,FN_sum,=[0,0,0,0]

    for image in input_image_list:
        args["image"]=os.getcwd()+f'/examples/exampleSet/photo/{image}'
        logging.debug(f'read image path:{args["image"]}\n')
        
        # load the known faces and embeddings
        tloadingStart=time.time()
        logging.info("[INFO] loading encodings...")
        data = pickle.loads(open(args["encodings"], "rb").read())
        # load the input image
        image = cv2.imread(args["image"])
        # resize image's width smaller than 1024px
        image=resize(image,image_acceptable_width)
        # convert it from BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tloadingEnd=time.time()


        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        logging.info("[INFO] recognizing faces...")

        # Encoding timing
        tEncodingStart=time.time()
        #extract image file's name
        imagepathTail=os.path.split(args["image"])[1]
        logging.debug(f'file : {imagepathTail}\n')
        label = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/testSetTable.csv',index_col=0)
        logging.debug(f'read csv : \n{label}\n')
        try:
            labelExt = label.loc[label['filename']==imagepathTail,:]
            logging.debug(f'labelExt : \n{labelExt}\n the sum of labelExt : {labelExt.iloc[:,1:5].sum(axis="columns").values.item()}\nindex of labelExt : {labelExt.index.values.item()}\n')
        except Exception as e:
            logging.info(f'{e.__class__.__name__} happen, check input Table or image file')
            break
        # setting number_of_times_to_upsample for face_locations
        locationsSize=1
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=locationsSize,
        model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        logging.debug(f'boxes : {len(boxes)}\n')

        # if face_location prediction is less than records in tabel => scale up locationsSize
        while len(boxes)<labelExt.iloc[:,1:5].sum(axis="columns").values.item():
            locationsSize+=1
            if locationsSize<3:
                logging.debug(f'face_location prediction is less than records =>\nlen of boxes: {len(boxes)}\nrecords : {labelExt.iloc[:,1:5].sum(axis="columns").values.item()}\nlocationsSize : {locationsSize}\n')
                boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=locationsSize,
                model=args["detection_method"])
                encodings = face_recognition.face_encodings(rgb, boxes)
                logging.debug(f'boxes_upsample : {len(boxes)}\n')
            else:
                show_image(image,boxes)
                reviseArgs = {}
                logging.info('input 1 if you want to revise, input 0 if you don\'t want to revise : ')
                reviseArgs['revise_or_not']=int(input())
                if reviseArgs['revise_or_not']==0 :
                    rm_img_ow_csv(label,imagepathTail)
                elif reviseArgs['revise_or_not']==1:
                    logging.debug(f'{reviseArgs}\n')
                    reviseArgs['hao'],reviseArgs['ywt'],reviseArgs['ford'],reviseArgs['ford']=revise_csv(label,imagepathTail,reviseArgs)
                    logging.debug(f'{reviseArgs}\n')
                    labelExt['hao']=reviseArgs['hao']
                    labelExt['ywt']=reviseArgs['ywt']
                    labelExt['ford']=reviseArgs['ford']
                    labelExt['unknown']=reviseArgs['unknown']
                break


        # if face_location prediction is more than records in tabel => recheck the picture and revise table
        while len(boxes)>labelExt.iloc[:,1:5].sum(axis="columns").values.item():
            logging.debug(f'\nlen of boxes: {len(boxes)}\nthe sum of labelExt : {labelExt.iloc[:,1:5].sum(axis="columns").values.item()}\n')
            show_image(image,boxes)
            reviseArgs = {}
            logging.info('input 1 if you want to revise, input 0 if you don\'t want to revise : ')
            reviseArgs['revise_or_not']=int(input())
            if reviseArgs['revise_or_not']==0 :
                rm_img_ow_csv(label,imagepathTail)
            elif reviseArgs['revise_or_not']==1 :
                logging.debug(f'{reviseArgs}\n')
                reviseArgs['hao'],reviseArgs['ywt'],reviseArgs['ford'],reviseArgs['ford']=revise_csv(label,imagepathTail,reviseArgs)
                logging.debug(f'{reviseArgs}\n')
                labelExt['hao']=reviseArgs['hao']
                labelExt['ywt']=reviseArgs['ywt']
                labelExt['ford']=reviseArgs['ford']
                labelExt['unknown']=reviseArgs['unknown']
            break



        # initialize the list of names for each face detected
        names = []
        tEncodingEnd=time.time()

        # timming Comparing time
        tCompareStart=time.time()
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

        # Counting the number of people which show in the image and be labeled in encoding set    
        count = Counter(names)
        logging.info(f'pridct names:{names},Count : {count}\n')


        # assuming the following sitution won't happen: 
        # the person_A who is labeled in encoding set shows up in the image but was missed by compare_faces(), and compare_faces() recognize some other people as person_A
        TP,FP,TN,FN=[0,0,0,0]
        for key in labelExt.columns[2:6]:
            print(count.get(key),labelExt[key].values.item(0), key,'\n============================================\n')
            # unknown is negative
            if key=='unknown':       
                # predictions of unknown people is more than records in table, the deviation is FN
                if count.get(key) and count.get(key)>=labelExt[key].values.item(0):
                    TN =labelExt[key].values.item(0)
                    FN = count.get(key)-labelExt[key].values.item(0)
                # predictions of unknown people is less than records in table => there are some unknown people is regarded as some labeled people(FP)
                elif count.get(key) and count.get(key)<labelExt[key].values.item(0):
                    TN = count.get(key)
                    FN = 0
            # other than unknown is positive 
            elif key!='unknown':
                #  predictions of labeled people is more than records in table, the deviation is FP
                if count.get(key) and count.get(key)>=labelExt[key].values.item(0):
                    TP = labelExt[key].values.item(0)
                    FP = count.get(key)-labelExt[key].values.item(0)
                # predictions of labeled people is less than records in table => there are some labeled people is regarded as some unknown people(FN)
                elif count.get(key) and count.get(key)<labelExt[key].values.item(0):
                    TP = count.get(key)
                    FP = 0
                
        logging.info(f'\n True Postive : {TP} \n False Postive : {FP} \n True Negtive : {TN} \n False Negtive {FN}\n')

        TP_sum+=TP;FP_sum+=FP;TN_sum+=TN;FN_sum+=FN
        
        tCompareEnd=time.time()

        # # resize image's width smaller than 1024px
        # hight,width=image.shape[:2]
        # logging.debug(f'image ori width:{width},image ori hight:{hight}\n')
        # if image.shape[1]>800:
        #     factor = 800/image.shape[1]
        #     logging.debug(f'resize fector:{factor}\n')
        #     width = 800
        #     hight = round(hight*factor)
        # logging.debug(f'image transfer width:{width},image transfer hight:{hight}\n')
        # image = cv2.resize(image,(width, hight), interpolation = cv2.INTER_CUBIC)

        # # loop over the recognized faces
        # for ((top, right, bottom, left), name) in zip(boxes, names):
        #     # draw the predicted face name on the image
        #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        #     y = top - 15 if top - 15 > 15 else top + 15
        #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #         0.75, (0, 255, 0), 2)

        # # show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # show the whole processing time
        logging.info(f'\n Loading time : {tloadingEnd-tloadingStart} \n Encoding time : {tEncodingEnd-tEncodingStart} \n Comapare time : {tEncodingEnd-tEncodingStart} \n Total recognize time : {tloadingEnd-tloadingStart+tEncodingEnd-tEncodingStart+tEncodingEnd-tEncodingStart}\n')
    logging.info(f'\nTP sum = {TP_sum} \n FP sum = {FP_sum} \n TN sum = {TN_sum} \n FN sum = {FN_sum}')

    conf_matrix=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/Confusion_matrix.csv',index_col=0)
    t=0.4
    conf_matrix=conf_matrix.set_index(['predict'],drop=True,append=True)
    conf_matrix.loc[(f'Tole:{t}','T'),'P']=TP_sum+conf_matrix.loc[(f'Tole:{t}','T'),'P']
    conf_matrix.loc[(f'Tole:{t}','F'),'P']=FP_sum+conf_matrix.loc[(f'Tole:{t}','F'),'P']
    conf_matrix.loc[(f'Tole:{t}','T'),'N']=TN_sum+conf_matrix.loc[(f'Tole:{t}','T'),'N']
    conf_matrix.loc[(f'Tole:{t}','F'),'N']=FN_sum+conf_matrix.loc[(f'Tole:{t}','F'),'N']
    conf_matrix.to_csv(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/Confusion_matrix.csv',index=True,header=True) 
    logging.debug('conf_M OW')
    print('recognize finish\n')
except Exception as e:
    print(e.__class__.__name__,'\n')
