import matplotlib
import matplotlib.pyplot as plt


x_value =[i  for i in range(0,10)]
y_value =[i for i in range(0,10)]

plt.plot(x_value,y_value)
plt.show()
plt.savefig('test.png')
