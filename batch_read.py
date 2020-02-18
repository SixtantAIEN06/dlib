import os
import subprocess as sp
import logging

image_file_list=sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/photo'))

image_acceptable_width=str(2560)

logging.basicConfig(level=logging.INFO,format='%(module)s--%(levelname)s--line : %(lineno)d\n%(message)s',filename='mylog_batch_read.txt')

stop_image_iteration=False
stop_tol_iteration=False

# adjust the range and interval of tolerance of face_compare
for _ in range(1,11):
    if not stop_tol_iteration :
        pass
    else:
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
                print(f"--------------tol : {tolerance}--batch : {_}-------------------------")
                input_image=image_file_list[_:_+batch_size]
                input_image=",".join(input_image)
                recog=sp.Popen(['python3','recognize_faces_image_loop_test.py','-e','encoding/encoding3.pickle','-i',input_image,'-iw',image_acceptable_width,"-t",tolerance],stdout=sp.PIPE)
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
                    logging.info(f'tolerance : {tolerance}\n{input_image} has been processed\n-------------------------------------------------\n')
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
                logging.debug('iteration finish\n')
            except StopIteration as e:
                logging.info(f'{e}\n{e.__class__.__name__,}')
                stop_image_iteration=True
                break
            except Exception as e:
                logging.info(f'-----------------other Exception-------------')
                logging.info(f'{e}\n{e.__class__.__name__,}')
                iteration_successful=True

