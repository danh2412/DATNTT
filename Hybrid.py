# line plot of time series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_error(y_true, y_pred):
    mean = mean_squared_error(y_true, y_pred)
    return np.sqrt(mean)


# load dataset
file_path = 'Weather_HCM.csv'
temperature = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])

# display first few rows
print(temperature.head(20))
Num_Test = 6
split_point = len(temperature) - Num_Test
dataset, validation = temperature[0:split_point], temperature[split_point: len(temperature)]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))

X = dataset.values
differenced = difference(X, 24)
d = 0
# fit model
model = ARIMA(differenced, order=(0, d, 2))
model_fit = model.fit(disp=0)

# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced) + Num_Test - 1
forecast = model_fit.predict(start=start_index, end=end_index)

# Residuals
residuals = np.round(model_fit.resid, decimals=2)
print('\nResiduals: %s\n' % residuals)

# invert the differenced forecast to something usable
predict = [x for x in np.round(X, decimals=2)]
day = 1
for yhat in forecast:
    inverted = np.round(inverse_difference(predict, yhat, 24), decimals=2)
    predict.append(inverted)
    day += 1

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

Num_Start = 24 + d
Num_Stop = 504
X = np.arange(Num_Stop).reshape(-1, 1)

X_training = X[Num_Start: Num_Stop - Num_Test]

svr_rbf = SVR(kernel='rbf', C=1, gamma=0.50)
svr_rbf.fit(X_training, residuals)
res_rbf = np.round(svr_rbf.predict(X[Num_Start:]), decimals=2)

print('\nResiduals predict: %s\n' % res_rbf[len(res_rbf) - Num_Test:])

plt.plot(np.arange(len(res_rbf) - Num_Test), residuals, color='darkorange', lw=2, label='Residuals')
plt.plot(np.arange(len(res_rbf)), res_rbf, color='navy', lw=2, label='Predict')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Residuals prediction')
plt.legend()
plt.show()

for i in range(Num_Test):
    predict[i + len(predict) - Num_Test] = np.round(predict[i + len(predict) - Num_Test] +
                                                    res_rbf[i + len(res_rbf) - Num_Test],
                                                    decimals=2)

mae_predict = mean_absolute_error(validation, predict[len(predict) - Num_Test:])
mse_predict = mean_squared_error(validation, predict[len(predict) - Num_Test:])
mape_predict = mean_absolute_percentage_error(validation, predict[len(predict) - Num_Test:])
rmse_predict = root_mean_squared_error(validation, predict[len(predict) - Num_Test:])
print('Actual:  %s' % np.array(validation).tolist())
print('Predict: %s' % np.array(predict[len(predict) - Num_Test:]).tolist())
print('MAE Predict = %f' % mae_predict)
print('MSE Predict = %f' % mse_predict)
print('MAPE Predict = %f' % mape_predict)
print('RMSE Predict = %f' % rmse_predict)

plt.plot(temperature.index[len(temperature) - Num_Test:], temperature['Temperature'][len(temperature) - 6:], '-o', color='darkorange', lw=2, label='data')
plt.plot(temperature.index[len(temperature) - Num_Test:], predict[len(predict) - 6:], '-s', color='navy', lw=2, label='Hybrid model')
# plt.plot(np.arange(len(validation)), validation, color='darkorange', lw=2, label='data')
# plt.plot(np.arange(len(validation)), predict[len(predict) - 7:], color='navy', lw=2, label='ARIMA model')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature prediction (Hybrid model)')
plt.legend()
plt.show()

