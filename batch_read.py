import os
import subprocess as sp

image_file_list=sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))+'/examples/exampleSet/photo'))

for _ in range(0,len(image_file_list),10):
    input_image=image_file_list[_:_+10]
    input_image=",".join(input_image)
    print(input_image)
    recog=sp.Popen(['python3','recognize_faces_image_loop_test.py','-e','encoding/encoding3.pickle','-i',input_image],stdout=sp.PIPE)
    res = recog.communicate()
    for line in res[0].decode(encoding='utf-8').split('\n'):
      print(line)
    # print(res[0].decode(encoding='utf-8').split('\n')[-3])