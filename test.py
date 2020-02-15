# import forImport_recognize_faces_image
# a = forImport_recognize_faces_image.readPara("encoding/encoding3.pickle",'examples/P_20200201_164242.jpg','cnn')
# @profile
# def my_func():
#     import gc
#     a=5
#     con=True
#     while con:
#         try:
#             if a>0:
#                 b=0
#             else:
#                 b=1
#             c=5/b
#             print(c)
#             d=5
#             con2=True
#             while con2:
#                 try:
#                     z=[1]*100
#                     print(hex(id(z)))
#                     if d>0:
#                         e=0
#                     else:
#                         e=1
#                     f=5/e
#                     print(f)
#                     con2=False
#                 except ZeroDivisionError as e:
#                     d-=1
#                     print(f'In {e} happen , now d = {d}')
#                     del z
#                     gc.collect()
#             con=False
#         except ZeroDivisionError as e:
#             a-=1
#             print(f'Out {e} happen , now a = {a}')
# if __name__=='__main__':
#     my_func()
#----------------------------------------------------------------------------
# import gc
# @profile
# def my_func():
#     a=[1]*100000
#     b=[20]*1000000
#     print(hex(id(a)))
#     print(hex(id([1]*100000)))
#     print(hex(id(b)))
#     print(hex(id([20]*1000000)))
#     del b
#     gc.collect()
#     print(hex(id(a)))
#     print(hex(id([1]*100000)))
#     # print(hex(id(b)))
#     print(hex(id([20]*
# if __name__=='__main__':
#     my_func()
#------------------------------------------------------------------------------

# print('out of my_func')
# def power2(x):
#     x=x**2
#     return x
# def my_func():
#     a=5
#     b=power2(a)
#     print(b)
# if __name__=='__main__':
#     my_func()
#------------------------------------------------------------------------------

# print('out of class test')
# class test():
#     def __init__(self):
#         pass
#     def power2(self,x):
#         x=x**2
#         return x
#     def my_func(self):
#         a=5
#         b=self.power2(a)
#         print(b)
# if __name__=='__main__':
#     A=test()
#     A.my_func()
#--------------------------------------------------------------------------------------------------
# try:
#     a=5/0
#     print(a)
# except Exception as e:
#     print(type(e),type(e).__name__,e.__class__.__name__,e.__class__.__qualname__)
#-------------------------------------------------------------------------------------------------
# cond=True
# a=0
# while cond :
#     for _ in range(10):
#         print(_)
#         if _>5:
#             print(f'{_}>5')
#             break
#         elif _>=8:
#             cond=False
#-------------------------------------------------------------------------------------------------
l1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
l2=[0,1,2,3,4,5,6,7,8,9]
bs=4
cond=True
while cond:
    for _ in range(0,len(l1)+1,bs):
        print(_)
        a=bs*(len(l1)//bs)
        print(a)
        if _==(bs*(len(l1)//bs)):
            cond=False
    






         