import os
import subprocess as sp
import logging
import re
import pandas as pd
import time

logging.basicConfig(level=logging.DEBUG,format='%(module)s--%(levelname)s--line : %(lineno)d\n%(message)s',filename='mylog_batch_read.txt')

encoding_file_list=sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/../encoding'))
#-----------------------------Turn on and ecit this to filter the encoding.pickle--------------------------------------- 
r=re.compile("\w+[^all]\_nj\w+\.pickle")
encoding_file_list=list(filter(r.match,encoding_file_list))
#-----------------------------Turn on and ecit this to filter the encoding.pickle---------------------------------------

image_file_list=sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/../testSet/photo'))
image_acceptable_width=str(2560)


#create and initialize Confusion_matrix.csv
init_conf_matrix=pd.DataFrame(columns=['encoding_set','num_jitters','tolerance','prediction','P','N'])
init_conf_matrix.to_csv(os.path.dirname(os.path.abspath(__file__))+'/Confusion_matrix.csv',index=False,header=True)
#create and initialize timing_log.csv
init_timing_log=pd.DataFrame(columns=['encoding_set','num_jitters','tolerance','loading','detecting','encoding','comparing','total'])
init_timing_log.to_csv(os.path.dirname(os.path.abspath(__file__))+'/timing_log.csv',index=False,header=True)

# initlize for loop
stop_encoding_iteration=False
stop_image_iteration=False
stop_tol_iteration=False

for encoding_set in encoding_file_list:
    logging.debug(f'encoding_file_list : {encoding_file_list}\n')
    if not stop_encoding_iteration :
        pass
    else:
        break
    # adjust the range and interval of tolerance of face_compare
    for _ in range(1,11):
        if not stop_tol_iteration :
            pass
        else:
            stop_encoding_iteration=True
            break
        tolerance=str(_/10)
        cond=True
        batch_size=10
        last_num=(batch_size*(len(image_file_list)//batch_size-1))
        Error_count=0
        # 
        for _ in range(0,len(image_file_list),batch_size):
            if not stop_image_iteration :
                pass
            else:
                stop_tol_iteration=True
                break
            iteration_successful=False
            while not iteration_successful:
                try :
                    print(f"--------------encoding set : {encoding_set}--tol : {tolerance}--batch : {_}-----------------")
                    input_image=image_file_list[_:_+batch_size]
                    input_image=",".join(input_image)
                    recog=sp.Popen(['python3','examining_para_loop.py','-e',f'../encoding/{encoding_set}','-i',input_image,'-iw',image_acceptable_width,"-t",tolerance],stdout=sp.PIPE)
                    res = recog.communicate()
                    
                    ##--------------------------uncomment this section to output all stdout or stderr----------------------
                    ## if the folloing code does't work, please add the "stderr=sp.PIPE" in the end of sp.Popen
                    # if res[1]==None :
                    #     print('--------------------------res[0]--------------------------\n')
                    #     for line in res[0].decode(encoding='utf-8').split('\n'):
                    #         print(line)
                    # else:
                    #     print('--------------------------res[1]--------------------------\n')
                    #     for line in res[1].decode(encoding='utf-8').split('\n'):
                    #         print(line)
                    # print('----------------------------res fin----------------------------\n')
                    ##--------------------------uncomment this section to output all stdout or stderr----------------------
                    
                    iteration_successful=False
                    if res[0].decode(encoding='utf-8').split('\n')[-3]:
                        status=res[0].decode(encoding='utf-8').split('\n')[-3].replace(" ","")
                        logging.info(f'recieve : {status}\n')
                        logging.info(f'encoding set : {encoding_set}, tolerance : {tolerance}\n{input_image} has been processed\n')
                    else:
                        status="unknown error please turn on the output all stdout or stderr"
                        logging.info(f'recieve : {status}\n')

                    if status=="RuntimeError" or status=="MemoryError":
                        error_message=res[0].decode(encoding='utf-8').split('\n')[-4]
                        error_location=res[0].decode(encoding='utf-8').split('\n')[-5]
                        logging.error(f'error message : {error_message}, happen in :{error_location}\n')
                        Error_count+=1
                        logging.debug(f'Error_count : {Error_count}\n')
                        if Error_count == 1:
                            image_acceptable_width=str(4096)
                            recog.kill()
                        elif Error_count == 2:
                            image_acceptable_width=str(2560)
                            recog.kill()
                        elif Error_count == 3:
                            image_acceptable_width=str(1920)
                            recog.kill()
                        elif Error_count == 4:
                            image_acceptable_width=str(1280)
                            recog.kill()
                        elif Error_count == 5:
                            image_acceptable_width=str(1024)
                            recog.kill()
                        elif Error_count == 6:
                            image_acceptable_width=str(960)
                            recog.kill()
                        elif Error_count == 7:
                            image_acceptable_width=str(800)
                            recog.kill()
                        elif Error_count == 8:
                            image_acceptable_width=str(640)
                            recog.kill()
                        elif Error_count == 9:
                            image_acceptable_width=str(480)
                            recog.kill()
                        elif Error_count == 10:
                            image_acceptable_width=str(320)
                            recog.kill()
                        else :
                            logging.error('please change your device, or check your code\n')
                            recog.kill()
                            cond=False
                            raise StopIteration
                        logging.error(f'{status} happened, now image_acceptable_width is {image_acceptable_width}\n')
                        continue
                    
                    if status=="ValueError":
                        raise StopIteration

                    recog.kill()
                    iteration_successful=True
                    logging.info('iteration finish-------------------------------------------------\n\n')
                except StopIteration as e:
                    logging.info(f'{e}\n{e.__class__.__name__,}')
                    stop_image_iteration=True
                    break
                except Exception as e:
                    logging.info(f'-----------------other Exception-------------')
                    logging.info(f'{e}\n{e.__class__.__name__,}')
                    iteration_successful=True

os.rename(os.path.dirname(os.path.abspath(__file__))+'/Confusion_matrix.csv',os.path.dirname(os.path.abspath(__file__))+f'/Confusion_matrix_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv')
os.replace(os.path.dirname(os.path.abspath(__file__))+f'/Confusion_matrix_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv',os.path.dirname(os.path.abspath(__file__))+f'/expriment_records/Confusion_matrix/Confusion_matrix_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv')

os.rename(os.path.dirname(os.path.abspath(__file__))+'/timing_log.csv',os.path.dirname(os.path.abspath(__file__))+f'/timing_log_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv')
os.replace(os.path.dirname(os.path.abspath(__file__))+f'/timing_log_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv',os.path.dirname(os.path.abspath(__file__))+f'/expriment_records/timing_log/timing_log_{time.strftime("%Y%m%d%H%M", time.localtime())}.csv')

os.rename(os.path.dirname(os.path.abspath(__file__))+'/mylog_batch_read.txt',os.path.dirname(os.path.abspath(__file__))+f'/mylog_batch_read_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt')
os.replace(os.path.dirname(os.path.abspath(__file__))+f'/mylog_batch_read_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt',os.path.dirname(os.path.abspath(__file__))+f'/expriment_records/log/mylog_batch_read_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt')
os.rename(os.path.dirname(os.path.abspath(__file__))+'/mylog_recog_loop.txt',os.path.dirname(os.path.abspath(__file__))+f'/mylog_recog_loop_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt')
os.replace(os.path.dirname(os.path.abspath(__file__))+f'/mylog_recog_loop_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt',os.path.dirname(os.path.abspath(__file__))+f'/expriment_records/log/mylog_recog_loop_{time.strftime("%Y%m%d%H%M", time.localtime())}.txt')
