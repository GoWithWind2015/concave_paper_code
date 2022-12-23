from matplotlib import pyplot as plt
import numpy as np
import util
from mpl_toolkits.mplot3d import Axes3D
import cv2
def return_tuple():
    a = ('a','b')
    a = a + ('c',)
    print(len(a))
    return a


my_a = return_tuple()
print(my_a)
print(len(my_a))
for i in my_a:
    print(i)
