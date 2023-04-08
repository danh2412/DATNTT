# line plot of time series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error  # , mean_squared_error


# create a differenced series
def difference(data, interval=1):
    diff = list()
    for i_diff in range(interval, len(data)):
        value = data[i_diff] - data[i_diff - interval]
        diff.append(value)
    return np.array(diff)


# invert differenced value
def inverse_difference(x, y, interval=1):
    return y + x[-interval]


def evaluate_svr_model(x_train, y_train, x_test, y_test, c_values, gamma_values):
    mean_error = np.zeros(shape=(len(c_values), len(gamma_values)))
    res_pred = np.zeros(shape=(len(c_values), len(gamma_values), len(x_test)))
    for c in c_values:
        c_index = c_values.tolist().index(c)
        for gamma in gamma_values:
            gamma_index = gamma_values.tolist().index(gamma)
            svr_rbf = SVR(kernel='rbf', C=c, gamma=gamma)
            svr_rbf.fit(x_train, y_train)
            res_pred[c_index, gamma_index] = np.round(svr_rbf.predict(x_test), decimals=2)

            mean_error[c_index, gamma_index] = mean_absolute_error(y_test, res_pred[c_index, gamma_index])
            index = (c_index * len(gamma_values)) + gamma_index
            print('%1.2f%%' % ((index * 100) / (len(gamma_values) * len(c_values))))
    min_mean = np.unravel_index(np.argmin(mean_error, axis=None), mean_error.shape)
    return mean_error, min_mean, res_pred[min_mean]


# load dataset
file_path = 'Weather_HCM.csv'
temperature = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])

# display first few rows
print(temperature.head(20))
Num_Test = 6
split_point = len(temperature) - Num_Test - 6
dataset, validation = temperature[0:split_point], temperature[split_point: len(temperature) - 6]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))

X = dataset.values
differenced = difference(X, 24)
p, d, q = 8, 0, 3
# fit model
model = ARIMA(differenced, order=(p, d, q))
model_fit = model.fit(disp=0)

# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced) + Num_Test - 1
forecast = model_fit.predict(start=start_index, end=end_index)

# Residuals
residuals = np.round(model_fit.resid, decimals=2)
# print('\nResiduals: %s\n' % residuals)

# invert the differenced forecast to something usable
predict = [x for x in np.round(X, decimals=2)]
for yhat in forecast:
    inverted = np.round(inverse_difference(predict, yhat, 24), decimals=2)
    predict.append(inverted)

mae_predict = mean_absolute_error(validation, predict[len(predict) - Num_Test:])
print('Actual:  %s' % np.array(validation).tolist())
print('Predict: %s' % np.array(predict[len(predict) - Num_Test:]).tolist())
print('MAE Predict = %f' % mae_predict)

# plt.plot(np.arange(len(temperature)), temperature['Temperature'], color='darkorange', lw=2, label='data')
# plt.plot(np.arange(len(temperature)), predict, color='navy', lw=2, label='ARIMA model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Temperature prediction (ARIMA model)')
# plt.legend()
# plt.show()

for i in range(Num_Test):
    res_test = np.round(validation.values[i] - predict[len(predict) - Num_Test + i], decimals=2)
    residuals = np.append(residuals, res_test)

Num_Start = 24 + d
Num_Stop = 498
X = np.arange(Num_Stop).reshape(-1, 1)

X_training = X[Num_Start: Num_Stop - Num_Test]
X_test = X[Num_Stop - Num_Test: Num_Stop]

C_range = np.array([1, 10, 100])
gamma_range = np.arange(0.001, 0.501, 0.001)

mae, best_score, res_rbf = evaluate_svr_model(X_training, residuals[: len(residuals) - Num_Test],
                                              X_test, residuals[len(residuals) - Num_Test:],
                                              C_range, gamma_range)

for i in range(Num_Test):
    predict[i + len(predict) - Num_Test] = np.round(predict[i + len(predict) - Num_Test] +
                                                    res_rbf[i],
                                                    decimals=2)

print('Actual:  %s' % residuals[len(residuals) - Num_Test:])
print('Predict: %s' % res_rbf)
print('C = %d | gamma = %1.2f' % (C_range[best_score[0]], gamma_range[best_score[1]]))
print('MAE Predict = %f' % mae[best_score])

plt.plot(np.arange(len(res_rbf)), residuals[len(residuals) - Num_Test:], color='darkorange', lw=2, label='Residuals')
plt.plot(np.arange(len(res_rbf)), res_rbf, color='navy', lw=2, label='Predict')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Residuals prediction')
plt.legend()
plt.show()

plt.plot(gamma_range, mae[0], 'r-', lw=2, label='C: 1')
plt.plot(gamma_range, mae[1], 'b--', lw=2, label='C: 10')
plt.plot(gamma_range, mae[2], 'g-.', lw=2, label='C: 100')
plt.xlabel('gamma')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (Hybrid model)')
plt.legend()
plt.show()
