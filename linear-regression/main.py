import pandas as pd 
import matplotlib.pyplot as plt

#data from kaggle: https://www.kaggle.com/datasets/andonians/random-linear-regression

data_train = pd.read_csv('train.csv')
data_train = data_train.dropna()
data_test = pd.read_csv('test.csv')

#visualization of data
#print(data_train)
#plt.scatter(data_train.x, data_train.y)
#plt.show()

#LOSS FUNCTION - won't be used to calculate a and b
# E=( y - ( ax + b ))^2 / n
def loss(a,b,data_points):
    error = 0.0
    for i in range(len(data_points)):
        x = data_points.iloc[i].x
        y = data_points.iloc[i].y
        error+= (y -(a*x + b))**2
    error = error / float(len(data_points))
    print(error)


#GRADINET DESCENT - this one will be used, it is the derivative we need
def gradient(a,b,data_points,L):
    a_grad = 0
    b_grad = 0
    n = len(data_points)

    for i in range(n):
        x = data_points.iloc[i].x
        y = data_points.iloc[i].y
        a_grad = x * (y - ( a*x + b ))
        b_grad = y - ( a*x + b )
    a_grad = a_grad * (-2) / n
    b_grad = b_grad * (-2) / n

    new_a = a - a_grad * L
    new_b = b - b_grad * L
    return new_a,new_b

a=0
b=0
L=0.0001
epochs = 2000

for i in range(epochs):
    a,b = gradient(a,b,data_train,L)
    if i%100 == 0:
        print(f"Epoch: {i}")
        loss(a,b,data_train)

print(a,b)
plt.scatter(data_train.x, data_train.y)
plt.plot(list(range(0,100)), [a*x+b for x in range(0,100)], color="black")
plt.show()