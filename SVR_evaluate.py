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


def evaluate_svr_model(x_train, y_train, x_test, y_test, c_values, gamma_values):
    mean_error = np.zeros(shape=(len(c_values), len(gamma_values)))
    y_pred = np.zeros(shape=(len(c_values), len(gamma_values), len(x_test)))
    for c in c_values:
        c_index = c_values.tolist().index(c)
        for gamma in gamma_values:
            gamma_index = gamma_values.tolist().index(gamma)
            svr_rbf = SVR(kernel='rbf', C=c, gamma=gamma)
            svr_rbf.fit(x_train, y_train)
            y_pred[c_index, gamma_index] = np.round(svr_rbf.predict(x_test), decimals=2)
            mean_error[c_index, gamma_index] = mean_absolute_error(y_test, y_pred[c_index, gamma_index])
            i = (c_index * len(gamma_values)) + gamma_index
            print('%1.2f%%' % ((i * 100) / (len(gamma_values) * len(c_values))))
    min_mean = np.unravel_index(np.argmin(mean_error, axis=None), mean_error.shape)
    return mean_error, min_mean, y_pred[min_mean]


file_path = 'Weather_HCM.csv'
data = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])
data.head(24)

temperature = np.array(data['Temperature'])  # 504 mau

Num_Start = 0  # Vi tri mau dau tien de training
Num_Stop = 492   # Vi tri mau cuoi cung de training
Num_Test = 6
X = np.arange(Num_Stop - Num_Start).reshape(-1, 1)

X_training = X[Num_Start: Num_Stop - Num_Test]
Y_training = temperature[Num_Start: Num_Stop - Num_Test]
X_test = X[Num_Stop - Num_Test: Num_Stop]
Y_test = temperature[Num_Stop - Num_Test: Num_Stop]

C_range = np.array([1, 10, 100])
gamma_range = np.arange(0.001, 0.501, 0.001)

mae, best_score, y_rbf = evaluate_svr_model(X_training, Y_training, X_test, Y_test, C_range, gamma_range)
mse = mean_squared_error(Y_test, y_rbf)
mape = mean_absolute_percentage_error(Y_test, y_rbf)
rmse = root_mean_squared_error(Y_test, y_rbf)

print('Actual:  %s' % Y_test)
print('Predict: %s' % y_rbf)
print('C = %d | gamma = %1.2f' % (C_range[best_score[0]], gamma_range[best_score[1]]))
print('MAE Predict = %f' % mae[best_score])
# print('MSE Predict = %f' % mse)
# print('MAPE Predict = %f' % mape)
# print('RMSE Predict = %f' % rmse)

plt.plot(gamma_range, mae[0], 'r-', lw=2, label='C: 1')
plt.plot(gamma_range, mae[1], 'b--', lw=2, label='C: 10')
plt.plot(gamma_range, mae[2], 'g-.', lw=2, label='C: 100')
plt.xlabel('gamma')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (SVR rbf model)')
plt.legend()
plt.show()

plt.plot(X_test * 24, Y_test, color='darkorange', lw=2, label='data')
plt.plot(X_test * 24, y_rbf, color='navy', lw=2, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Temperature prediction (SVR rbf model)')
plt.legend()
plt.show()
