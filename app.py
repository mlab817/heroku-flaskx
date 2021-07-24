# app.py
from flask import Flask, request, jsonify, render_template, json
from models.ltt import run_ols
from models.cagr import run_cagr
from models.arima import run_arima

import pandas as pd
import simplejson

app = Flask(__name__)


# A welcome message to test our server
@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    if request.method == 'GET':
        response = render_template('upload.html')
    #
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        # validate the uploaded files if it contains 5 columns

        df = pd.read_csv(uploaded_file,
                         thousands=',',
                         header='infer', )

        if len(df.axes[1]) != 5:
            return 'File must have 5 columns: year, production, area harvested, yield, and per capita. Your file has %s' % str(
                len(df.axes[1]))

        prepared_data = prepare_data(df)

        ols_result = run_ols(data=prepared_data, y_var='yield', x_var='year', log=True)

        if request.is_json:
            response = simplejson.dumps(ols_result, ignore_nan=True)
        else:
            response = simplejson.dumps(ols_result, ignore_nan=True)

    return response  # render_template('upload.html', shape=(32, 3))


# TODO: Make this prepare data dynamic
def prepare_data(data=pd.DataFrame()):
    """
    Prepare uploaded data for processing by defining the correct data types

    :param data:
    :return: prepared data
    """
    data.columns = ["year", "production", "area_harvested", "yield", "per_capita"]

    # convert data to float as sometimes the data is not of correct type (e.g. string instead of float)
    data["production"] = pd.to_numeric(data["production"])
    data["area_harvested"] = pd.to_numeric(data["area_harvested"])
    data["yield"] = pd.to_numeric(data["yield"])
    data["per_capita"] = pd.to_numeric(data["per_capita"])

    # verify data types
    data.describe()

    return data


if __name__ == '__main__':
    app.debug = True
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
