# line plot of time series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# create a differenced series
def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
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
# dataset.to_csv('dataset.csv')
# validation.to_csv('validation.csv')
# line plot of dataset
# print(np.array(temperature['Temperature']))
# temperature.plot()
# plt.show()
X = dataset.values
differenced = difference(X, 24)
# fit model
model = ARIMA(differenced, order=(8, 0, 3))
model_fit = model.fit()
# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced) + Num_Test - 1
forecast = model_fit.predict(start=start_index, end=end_index)
res = np.round(model_fit.resid, decimals=2)
print('res %s len %d' % (res, len(res)))

# invert the differenced forecast to something usable
predict = [x for x in np.round(X, decimals=2)]
for yhat in forecast:
    inverted = np.round(inverse_difference(predict, yhat, 24), decimals=2)
    predict.append(inverted)

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
plt.plot(temperature.index[len(temperature) - Num_Test:], predict[len(predict) - 6:], '-s', color='navy', lw=2, label='ARIMA model')
# plt.plot(np.arange(len(temperature)), temperature, color='darkorange', lw=2, label='data')
# plt.plot(np.arange(len(validation)), predict[len(predict) - Num_Test:], color='navy', lw=2, label='ARIMA model')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature prediction (ARIMA model)')
plt.legend()
plt.show()
