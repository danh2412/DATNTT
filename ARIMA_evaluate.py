# line plot of time series
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, yy = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, yy))


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


def evaluate_arima_model(training_data, validation_data, p_values, d_values, q_values):
    mean_error = np.zeros(shape=(len(p_values), len(d_values), len(q_values)))
    y_pred = np.zeros(shape=(len(p_values), len(d_values), len(q_values), len(validation_data)))
    for d in d_values:
        d_index = d_values.tolist().index(d)
        for p in p_values:
            p_index = p_values.tolist().index(p)
            for q in q_values:
                q_index = q_values.tolist().index(q)

                try:
                    differenced = difference(training_data, 24)
                    # fit model
                    model = ARIMA(differenced, order=(p, d, q))
                    model_fit = model.fit(disp=0)
                    # one-step out of sample forecast
                    start_index = len(differenced)
                    end_index = len(differenced) + len(validation_data) - 1
                    forecast = model_fit.predict(start=start_index, end=end_index)

                    # invert the differenced forecast to something usable
                    predict = [x for x in training_data]
                    for yhat in forecast:
                        inverted = np.round(inverse_difference(predict, yhat, 24), decimals=2)
                        predict.append(inverted)

                    y_pred[p_index, d_index, q_index] = predict[len(predict) - len(validation_data):]
                    mean_error[p_index, d_index, q_index] = mean_absolute_error(validation_data,
                                                                                y_pred[p_index, d_index, q_index])
                    print('(p,d,q): (%s,%s,%s): MAE=%f' % (p, d, q, mean_error[p_index, d_index, q_index]))
                    # print('p = %d | d = %d | q = %d' % (p, d, q))
                except:
                    mean_error[p_index, d_index, q_index] = 10
                    continue
    min_mean = np.unravel_index(np.argmin(mean_error, axis=None), mean_error.shape)
    return mean_error, min_mean, y_pred[min_mean]


# load dataset
file_path = 'Weather_HCM.csv'
temperature = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])
# temperature = temperature[: len(temperature) - 6]
# display first few rows
Num_Test = 6
split_point = len(temperature) - Num_Test - 6
dataset, validation = temperature[0:split_point], temperature[split_point: len(temperature) - 6]
# dataset.to_csv('dataset.csv')
# validation.to_csv('validation.csv')
# line plot of dataset
# print(np.array(temperature['Temperature']))
# temperature.plot()
# plt.show()

p_range = np.arange(11)  # 0 -> 10
d_range = np.arange(3)  # 0 -> 2
q_range = np.arange(11)  # 0 -> 10
warnings.filterwarnings("ignore")
mae, best_score, y_predict = evaluate_arima_model(dataset.values, validation.values, p_range, d_range, q_range)

print('Actual:  %s' % validation.values.reshape(1, -1))
print('Predict: %s' % y_predict)
print('(p, d, q) = (%d, %d, %d)' % (p_range[best_score[0]], d_range[best_score[1]], q_range[best_score[2]]))
print('MAE Predict = %f' % mae[best_score])

plt.subplot(131)
plt.imshow(mae[:, 0, :],
           interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0., midpoint=0.5))
plt.xlabel('q')
plt.ylabel('p')
plt.colorbar()
plt.xticks(np.arange(len(q_range)), q_range)
plt.yticks(np.arange(len(p_range)), p_range)
plt.title('MAE (ARIMA model) (d = 0)')

plt.subplot(132)
plt.imshow(mae[:, 1, :],
           interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0., midpoint=0.5))
plt.xlabel('q')
plt.ylabel('p')
plt.colorbar()
plt.xticks(np.arange(len(q_range)), q_range)
plt.yticks(np.arange(len(p_range)), p_range)
plt.title('MAE (ARIMA model) (d = 1)')

plt.subplot(133)
plt.imshow(mae[:, 2, :],
           interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0., midpoint=0.5))
plt.xlabel('q')
plt.ylabel('p')
plt.colorbar()
plt.xticks(np.arange(len(q_range)), q_range)
plt.yticks(np.arange(len(p_range)), p_range)
plt.title('MAE (ARIMA model) (d = 2)')
plt.show()

# plt.plot(np.arange(len(temperature)), temperature['Temperature'], color='darkorange', lw=2, label='data')
# plt.plot(np.arange(len(temperature)), predict, color='navy', lw=2, label='ARIMA model')
# # plt.plot(np.arange(len(validation)), validation, color='darkorange', lw=2, label='data')
# # plt.plot(np.arange(len(validation)), predict[len(predict) - 7:], color='navy', lw=2, label='ARIMA model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Temperature prediction (ARIMA model)')
# plt.legend()
# plt.show()
