import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simplejson
import statsmodels.api as sm
import requests
import uuid
import os

from models.mape import mape

url = 'http://nfcqs.example.com/api/handle_ltt'
url2 = 'http://nfcqs.example.com/api/handle_image_upload'


# the log parameter determines whether the model should be run as Log Time Trend
def run_ols(data=None, y_var="", x_var="year", log=False):
    """
    Function to run log time trend
    """
    # print(data)
    # create local copy of dataframe
    df = data
    # if log is turned on, generate a log value of the variable
    if log:
        # generate a new variable with name ln_var
        final_y_var = 'ln_%s' % y_var
        df[final_y_var] = np.log(df[y_var]).round(4)
    else:
        final_y_var = y_var

    x = df[x_var]

    X = sm.add_constant(df[x_var])
    y = df[final_y_var]
    sig = 0.25
    # ynorm = y + sig * np.random.normal(size=len(df))
    result = sm.OLS(y, X).fit()

    # do in-sample predictions
    ypred = result.predict(X)

    x1n = np.arange(2019, 2050, 1)
    Xnew = sm.add_constant(x1n)
    ynewpred = result.predict(Xnew)
    # print('Summary: ', result.summary())

    # let's stack up the results to generate a prediction df
    prediction = pd.DataFrame()
    prediction['year'] = np.hstack((x, x1n))
    prediction['ln_prediction'] = np.hstack((ypred.round(4), ynewpred.round(4)))
    prediction = pd.concat([prediction, df[final_y_var]], ignore_index=True, axis=1)
    prediction.columns = ['year', 'ln_prediction', 'ln_actual']
    prediction['abs_prediction'] = round(np.exp(prediction.ln_prediction), 4)
    prediction['abs_actual'] = round(np.exp(prediction.ln_actual), 4)

    response_json = dict()
    response_json['data'] = prediction.to_dict(orient="records")
    response_json['regression_params'] = result.params.to_dict()
    response_json['mape'] = mape(prediction.ln_actual, prediction.ln_prediction)
    response_json['y_variable'] = y_var
    response_json['x_variable'] = x_var
    response_json['model'] = 'Log Time Trend'

    # generate graph
    fig, ax = plt.subplots()
    #     ax.plot(x, y, 'o', label="Data")
    #     ax.plot(x, ynorm, 'b-', label="True")
    ax.plot(prediction.year, prediction.abs_prediction, 'b-', label="OLS prediction")
    ax.plot(prediction.year, prediction.abs_actual, 'o', label="Actual Value")
    ax.legend(loc="best")
    filename = 'ltt_' + str(uuid.uuid4()) + '.png'
    fig_path = os.path.join('static/images/', filename)
    fig.savefig(fig_path)

    try:
        print(simplejson.dumps(response_json, ignore_nan=True))
        resp = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            # json=simplejson.dumps(response_json, ignore_nan=True)
            # json=response_json
            json=format_response(response_json)
        )
        resp1 = requests.post(url2, headers={'Content-Type': 'multipart/form-data'},
                              files={'file': open(fig_path, 'rb').read()})
        print(resp)
        print(resp1)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)

    return response_json


def format_response(raw_data=None):
    """
    Convert response to dict() without the ignoring NaN to return json-friendly data
    """
    if raw_data is None:
        raw_data = dict()
    return simplejson.loads(simplejson.dumps(raw_data, ignore_nan=True))
