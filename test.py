# import forImport_recognize_faces_image
# a = forImport_recognize_faces_image.readPara("encoding/encoding3.pickle",'examples/P_20200201_164242.jpg','cnn')
@profile
def my_func():
    import gc
    a=5
    con=True
    while con:
        try:
            if a>0:
                b=0
            else:
                b=1
            c=5/b
            print(c)
            d=5
            con2=True
            while con2:
                try:
                    z=[1]*100
                    print(hex(id(z)))
                    if d>0:
                        e=0
                    else:
                        e=1
                    f=5/e
                    print(f)
                    con2=False
                except ZeroDivisionError as e:
                    d-=1
                    print(f'In {e} happen , now d = {d}')
                    del z
                    gc.collect()
            con=False
        except ZeroDivisionError as e:
            a-=1
            print(f'Out {e} happen , now a = {a}')

if __name__=='__main__':
    my_func()
         