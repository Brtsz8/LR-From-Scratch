# Linear Regression with Gradient Descent from scratch 
This repository contains a simple implementation of linear regression using gradient descent. The project trains a model to fit a linear function to a dataset from Kaggle.

📂Dataset: https://www.kaggle.com/datasets/andonians/random-linear-regression

This repository serves as a personal reference for implementing gradient descent in linear regression. Others are welcome to use or improve it as needed.

Some important concepts: 

Loss:
```
# E=( y - ( ax + b ))^2 / n
def loss(a,b,data_points):
    error = 0.0
    for i in range(len(data_points)):
        x = data_points.iloc[i].x
        y = data_points.iloc[i].y
        error+= (y -(a*x + b))**2
    error = error / float(len(data_points))
    print(error)
```

Gradient:
```
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
```