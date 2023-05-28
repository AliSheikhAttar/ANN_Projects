import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor





x = np.arange(0.1,0.85,0.05)
print(x)
out = [0.4, 0.35, 0.3, 0.55, 0.85, 0.83, 0.9, 0.97, 0.8, 1, 0.85, 0.84, 0.87, 0.55, 0.35]  
x_train, x_test, y_train, y_test = train_test_split(x, out)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')
plt.show()


x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


mlp = MLPRegressor(
    hidden_layer_sizes=[10,30,10],
    max_iter=3000, #2000 best 3000 second
    tol=0,
)

# train network
mlp.fit(x_train,y_train)

# test
predictions = mlp.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("{mse}")
plt.plot(x_test, predictions, 'ro')
plt.show()
plt.show()