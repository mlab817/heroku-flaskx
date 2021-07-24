from pmdarima.arima import auto_arima
from models.mape import mape

import pandas as pd
import numpy as np


def run_arima(df=pd.DataFrame(), column_name=''):
    y = df[column_name]
    x = df.year
    arima_model = auto_arima(y)

    # compute predictions for in and out sample
    in_sample_prediction = arima_model.predict_in_sample()
    out_sample_prediction = arima_model.predict(n_periods=31, return_conf_int=True, alpha=0.05)

    # create a new data frame for in-sample and out-of-sample prediction
    prediction = pd.DataFrame(columns=['year', 'prediction'])
    prediction['year'] = np.hstack([x, np.arange(2019, 2050, 1)])
    prediction['prediction'] = np.hstack([in_sample_prediction, out_sample_prediction[0]])

    mape_value = mape(in_sample_prediction, y)

    response_json = dict()
    response_json['data'] = prediction.to_dict(orient="records")
    response_json['mape'] = mape_value
    response_json['result'] = arima_model.params

    # plt.figure(figsize=(8, 5))
    # # plt.plot(train, label="Training")
    # plt.plot(y, 'o', label="Actual")
    # plt.plot(prediction.prediction, label="Prediction")
    # plt.title('ARIMA: Actual vs Predicted')
    # plt.legend(loc="best")
    # plt.show()
    #
    # plt.savefig(columnName + '_arima.jpg', dpi=300)
    return response_json
