import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_error(y_true, y_pred):
    mean = mean_squared_error(y_true, y_pred)
    return np.sqrt(mean)


file_path = 'Weather_HCM.csv'
data = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])
data.head(24)

temperature = np.array(data['Temperature'])  # 528 mau

Num_Start = 0  # Vi tri mau dau tien de training
Num_Stop = 504   # Vi tri mau cuoi cung de training
Num_Test = 6
X = np.arange(Num_Stop - Num_Start).reshape(-1, 1)

X_training = X[Num_Start: Num_Stop - Num_Test]
Y_training = temperature[Num_Start: Num_Stop - Num_Test]

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.32)
svr_rbf.fit(X_training, Y_training)
y_rbf = np.round(svr_rbf.predict(X[Num_Start: Num_Stop]), decimals=2)

y_rbf_train = y_rbf[0: len(y_rbf) - Num_Test]
y_rbf_pred = y_rbf[len(y_rbf) - Num_Test: len(y_rbf)]

mae_predict = mean_absolute_error(temperature[Num_Stop - Num_Test: Num_Stop], y_rbf_pred)
mse_predict = mean_squared_error(temperature[Num_Stop - Num_Test: Num_Stop], y_rbf_pred)
mape_predict = mean_absolute_percentage_error(temperature[Num_Stop - Num_Test: Num_Stop], y_rbf_pred)
rmse_predict = root_mean_squared_error(temperature[Num_Stop - Num_Test: Num_Stop], y_rbf_pred)

print('Actual:  %s' % (temperature[Num_Stop - Num_Test: Num_Stop]))
print('Predict: %s' % (y_rbf[len(y_rbf) - Num_Test: len(y_rbf)]))
print('MSE Predict = %f' % mse_predict)
print('MAE Predict = %f' % mae_predict)
print('MAPE Predict = %f' % mape_predict)
print('RMSE Predict = %f' % rmse_predict)
# print(data.index[Num_Stop - Num_Test:])
# plt.scatter(X[Num_Start: Num_Stop], temperature[Num_Start: Num_Stop], color='darkorange', label='data')
plt.plot(data.index[Num_Stop - Num_Test:], temperature[Num_Stop - Num_Test:], '-o', color='darkorange', lw=2, label='data')
plt.plot(data.index[Num_Stop - Num_Test:], y_rbf[len(y_rbf) - Num_Test: len(y_rbf)], '-s', color='navy', lw=2, label='RBF model')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature prediction (SVR rbf model)')
plt.legend()
plt.show()
