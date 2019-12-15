import matplotlib.pyplot as plt 
import numpy as np 
width = 0.3
height_1 = np.arange(0,1,0.2)
height_2 = np.arange(1,2,0.2)

x_1 = np.arange(5)
x_2 = [i+width for i in x_1]
plt.bar(x=x_1,height=height_1,width=width,color='r',label = 'with_out_rl')
plt.bar(x=x_2,height=height_2,width=width,color='g',label = 'with_rl')
plt.legend()
plt.show()