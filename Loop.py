import pandas as pd
import csv
import numpy as np
# from sklearn.svm import SVR
from urllib import request
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def preprocess_data(url_link, save_path):
    status_t = 1
    file_path = 'get_file.csv'
    request.urlretrieve(url_link, file_path)
    print("Test")
    # get last index in database    
    data = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, index_col=[0])
    index = int(data.tail(1)['entry_id'].values)

    # get last index in save file
    with open("last_index.txt", "r") as f:
        last_index = int(f.readlines()[-1])

    if index > last_index:
        # get last data in save file
        save_data = pd.read_csv(save_path, delimiter=',', header=0, skipinitialspace=True)
        last_data = save_data.tail(1).values[-1][1]
        # print('last_data %d' % last_data)

        for i in range(last_index, index):
            with open(file_path, "r") as f, open(save_path, "a") as g:
                last_line = f.readlines()[i + 1].strip().split(",")[0:3:2]
                print('last_line %s' % last_line[1])
                if float(last_line[1]) != last_data:
                    #print('Database %f | Last_data %f' % (float(last_line[1]), last_data))
                    status_t = 0
                    c = csv.writer(g)
                    c.writerow(last_line)
                    last_data = last_line[1]

        with open("last_index.txt", "w") as g:
            g.write(str(index))
            g.close()

    return status_t


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


while True:
    save_file = 'Humidity_HCM.csv'
    url = "https://api.thingspeak.com/channels/732264/fields/2.csv?results=200"
    status = preprocess_data(url, save_file)
    if status == 0:
        humidity = pd.read_csv(save_file, delimiter=',', header=0, skipinitialspace=True, index_col=[0])

        Num_Test = 1
        split_point = len(humidity) - Num_Test
        dataset, validation = humidity[0:split_point], humidity[split_point: len(humidity)]

        X = dataset.values
        differenced = difference(X, 24)
        # fit model
        model = ARIMA(differenced, order=(1, 0, 1))
        model_fit = model.fit()
        # one-step out of sample forecast
        start_index = len(differenced)
        end_index = len(differenced) + Num_Test - 1
        forecast = model_fit.predict(start=start_index, end=end_index)

        # invert the differenced forecast to something usable
        predict = [x for x in np.round(X, decimals=2)]
        for yhat in forecast:
            inverted = np.round(inverse_difference(predict, yhat, 24), decimals=2)
            predict.append(inverted)

        time = (80 - predict[len(predict) - 1]) / 2
        if time < 0:
            time = 0

        send_data = request.urlopen('https://api.thingspeak.com/update?api_key=UIPTABPX07XNJVNE&field1='
                                    + str(int(np.round(predict[len(predict) - 1], 2)))
                                    + '&field2=' + str(int(1)))
        print('https://api.thingspeak.com/update?api_key=UIPTABPX07XNJVNE&field1='
              + str(float(np.round(predict[len(predict) - 1], 2)))
              + '&field2=' + str(int(time)))

    else:
        # Sensor_Data = np.append(Sensor_Data, y_rbf[len(y_rbf) - 1])
        # Num = np.arange(len(Sensor_Data)).reshape(-1, 1)
        # plt.plot(Num[0: len(Sensor_Data)], Sensor_Data, color='darkorange', lw=2, label='data')
        # plt.plot(Num, y_rbf, color='navy', lw=2, label='RBF model')
        # plt.xlabel('data')
        # plt.ylabel('target')
        # plt.title('Support Vector Regression')
        # plt.legend()
        # if plt.waitforbuttonpress(60) is True:
        #     break
        # else:
        #     plt.clf()
        print("okokokokoko")
        continue
