import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as li
import sklearn.metrics as met



np.random.seed(3)

n1 = 4000
n2 = 3
X = np.zeros((n1,n2))
Y = np.zeros((n1,1))


noise = np.random.uniform(-0.2,0.2)

for i in range(n1):
    X [i, :] = np.random.uniform(-1,1,n2)
    Y [i] = 0.1*np.sin(X[i ,0]) + 0.3*np.sin(X[i,1]) + 0.8*np.sin(X[i,2]) + noise
# by reduce the Coefficients by 0.1 the error reduced by 0.01

model = li.LinearRegression()

model.fit(X,Y)

o = model.predict(X)

mse = met.mean_squared_error(Y,o)
print(f"the mean squared error is {mse}")
plt.scatter(Y[:] , o[:] , s = 10)
plt.xlabel('target')
plt.ylabel('predicted')
plt.plot([-3,3],[-3,3] , c = 'red',label = 'y=x')
plt.legend()
plt.show()


t = np.linspace(-2, 2, 600)

a = 0.1*np.sin(t) + 0.3*np.sin(t) + 0.8*np.sin(t)

t1 = np.zeros((len(t),n2))
for i in range(len(t)):
    t1[i,0] = t[i]
    t1[i,1] = t[i]
    t1[i,2] = t[i]

b = model.predict(t1)


plt.plot(t, a, 'r') 
plt.plot(t, b, 'b') 
plt.show()


