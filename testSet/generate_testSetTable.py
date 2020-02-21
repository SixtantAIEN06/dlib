import os
import pandas as pd
import re
import logging 

logging.basicConfig(level=logging.INFO)
logging.debug(os.path.dirname(os.path.abspath(__file__))+'\n')
logging.info(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/photo'))
data=[{'filename':f'{filename}','ywt':0,'hao':0,'ford':0,'unknown':0,'DOD':0} for filename in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/photo'))]
test_photo_property = pd.DataFrame(data)
test_photo_property.loc[test_photo_property['filename'].str.match(r"\w+ford|ford"),'ford']=1
test_photo_property.loc[test_photo_property['filename'].str.match(r"\w+hao|hao"),'hao']=1
test_photo_property.loc[test_photo_property['filename'].str.match(r"\w+ywt|ywt"),'ywt']=1
logging.debug(f'workingDir : {os.getcwd()} , filename : {__file__} , dirname : {os.path.dirname(__file__)} , abspath : {os.path.abspath(__file__)} , base : {os.path.basename(__file__)} , dir(abs) : {os.path.dirname(os.path.abspath(__file__))}')

# test_photo_property.to_csv(os.path.dirname(os.path.abspath(__file__))+'/testSetTable.csv')
