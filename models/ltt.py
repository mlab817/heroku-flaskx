import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


from models.mape import mape


# the log parameter determines whether the model should be run as Log Time Trend
def run_ols(data=None, y_var="", x_var="year", log=False):
    print(data)
    # create local copy of dataframe
    df = data
    # if log is turned on, generate a log value of the variable
    if log:
        # generate a new variable with name ln_var
        final_y_var = 'ln_%s' % y_var
        df[final_y_var] = np.log(df[y_var])
    else:
        final_y_var = y_var

    x = df[x_var]
    # print(x)
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
    prediction['prediction'] = np.hstack((ypred, ynewpred))
    prediction = pd.concat([prediction, df[final_y_var]], ignore_index=True, axis=1)
    prediction.columns = ['year', 'prediction', 'actual']

    response_json = dict()
    response_json['data'] = prediction.to_dict(orient="records")
    response_json['regression_params'] = result.params.to_dict()
    response_json['mape'] = mape(prediction.actual, prediction.prediction)

    fig, ax = plt.subplots()
    #     ax.plot(x, y, 'o', label="Data")
    #     ax.plot(x, ynorm, 'b-', label="True")
    ax.plot(prediction.year, prediction.prediction, 'b-', label="OLS prediction")
    ax.plot(prediction.year, prediction.actual, 'o', label="Actual Value")
    ax.legend(loc="best")

    return response_json
