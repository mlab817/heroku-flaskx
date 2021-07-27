# app.py
from flask import Flask, request, render_template
from pmdarima.arima import auto_arima

import pandas as pd
import simplejson
import statsmodels.api as sm
import requests
import numpy as np
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
        commodity_data = request.files.get('commodity_data')
        population_data = request.files.get('population_data')
        population = request.form.get('population')
        population_year = request.form.get('population_year')
        per_capita = request.form.get('per_capita')
        crop_type = request.form.get('crop_type')
        conversion_rate = request.form.get('conversion_rate')
        crop_id = request.form.get('crop_id')

        results = run_models(
            commodity_data=commodity_data,
            population_data=population_data,
            population=int(population),
            population_year=int(population_year),
            per_capita=float(per_capita),
            crop_type=str(crop_type),
            conversion_rate=float(conversion_rate)
        )

        data = results.to_dict(orient='records')

        response = dict()
        response['crop_id'] = crop_id
        response['data'] = simplejson.loads(simplejson.dumps(data, ignore_nan=True))

        print(response)

        try:
            url = 'http://nfcqs.example.com/api/handle_save_model_results'

            resp = requests.post(
                url,
                headers={'Content-Type': 'application/json'},
                # json=simplejson.dumps(response_json, ignore_nan=True)
                # json=response_json
                json=response
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)

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


# the log parameter determines whether the model should be run as Log Time Trend
def run_ltt(data=None, crop_type='', conversion_rate=0.0):
    """
    Function to calculate log time trend for production
    """
    # duplicate data
    df = data
    # set year as index
    df.set_index('year')

    x = sm.add_constant(df.year)
    x1n = np.arange(df.year.max() + 1, 2050, 1)
    Xnew = sm.add_constant(x1n)

    prediction = pd.DataFrame()
    prediction['year'] = np.hstack((df.year, x1n))

    if crop_type.lower() == 'crop':
        # use log on yield and area
        df['ln_area'] = np.log(df['area_harvested'])
        df['ln_yield'] = np.log(df['yield'])
        # run the models
        area_model = sm.OLS(df.ln_area, x).fit()
        yield_model = sm.OLS(df.ln_yield, x).fit()
        # predict yield and area using log values
        in_sample_area = area_model.predict(x)
        in_sample_yield = yield_model.predict(x)
        # multiply the area and yield to get production
        # reverse log the predicted production
        out_sample_area = area_model.predict(Xnew)
        out_sample_yield = yield_model.predict(Xnew)
        # combine to get final raw df
        prediction['predicted_area'] = np.hstack((in_sample_area, out_sample_area))
        prediction['predicted_yield'] = np.hstack((in_sample_yield, out_sample_yield))
        # computed predicted production
        prediction['ltt_production'] = round(np.exp(prediction.predicted_area) * np.exp(prediction.predicted_yield), 8)
        # prediction['ltt_production'] = prediction.ltt_production * conversion_rate / 100
    else:
        df['ln_production'] = np.log(df['production'])
        production_model = sm.OLS(df.ln_production, x).fit()
        in_sample_production = production_model.predict(x)
        out_sample_production = production_model.predict(Xnew)
        prediction['predicted_production'] = np.hstack((in_sample_production, out_sample_production))
        prediction['ltt_production'] = round(np.exp(prediction.predicted_area) * np.exp(prediction.predicted_yield), 8)

    return prediction[['year', 'ltt_production']]


def run_cagr(data=None, crop_type='crop', conversion_rate=0.0):
    """
    Function to run compounded annual growth rate
    """
    df = data
    # set year as index
    df.set_index('year')

    if crop_type.lower() == 'crop':
        # get the first and last values of the variable
        start_value_area = df['area_harvested'].iloc[0]
        end_value_area = df['area_harvested'].iloc[-1]
        cagr_area = (end_value_area / start_value_area) ** (1 / len(df)) - 1

        start_value_yield = df['yield'].iloc[0]
        end_value_yield = df['yield'].iloc[-1]
        cagr_yield = (end_value_yield / start_value_yield) ** (1 / len(df)) - 1

        # create a new dataframe to compare the prediction and actual values
        prediction = pd.DataFrame(
            [[df.year.min() + i, start_value_area * (1 + cagr_area) ** i] for i in range(len(df) + (2049 - df.year.max()))],
            columns=['year', 'predicted_area'])
        prediction2 = pd.DataFrame([[df.year.min() + i, start_value_yield * (1 + cagr_yield) ** i] for i in
                                    range(len(df) + (2049 - df.year.max()))], columns=['year', 'predicted_yield'])
        prediction = pd.merge(prediction, prediction2, on='year', how='left')
        prediction['cagr_production'] = round(prediction.predicted_area * prediction.predicted_yield, 6)
        prediction['cagr_production'] = prediction.cagr_production * conversion_rate / 100
    else:
        start_value_production = df['production'].iloc[0]
        end_value_production = df['production'].iloc[-1]
        cagr_production = (end_value_production / start_value_production) ** (1 / len(df)) - 1

        # create a new dataframe to compare the prediction and actual values
        prediction = pd.DataFrame(
            [[df.year.min() + i, start_value_production * (1 + cagr_production) ** i] for i in
             range(len(df) + (2049 - df.year.max()))],
            columns=['year', 'cagr_production'])
        prediction['cagr_production'] = prediction.cagr_production * conversion_rate / 100

    #     predicted_yield =
    #     prediction = pd.merge(predicted_area, predicted_yield)
    prediction.head()
    #     prediction['predicted_production'] = df.predicted_area * df.predicted_yield
    #     print(prediction.head())

    return prediction[['year', 'cagr_production']]


def run_arima(data=None, conversion_rate=0.0):
    """
    Project production using ARIMA
    """
    df = data
    y = df['production']
    x = df.year
    arima_model = auto_arima(y)

    in_sample_prediction = arima_model.predict_in_sample()
    out_sample_prediction = arima_model.predict(n_periods=31, return_conf_int=True, alpha=0.05)
    prediction = pd.DataFrame(columns=['year', 'predicted_production'])
    prediction['year'] = np.hstack([x, np.arange(df.year.max() + 1, 2050, 1)])
    prediction['arima_production'] = np.hstack([in_sample_prediction, out_sample_prediction[0]])
    prediction['arima_production'] = prediction.arima_production * conversion_rate / 100
    return prediction[['year', 'arima_production']]


def run_models(commodity_data=None,
               population_data=None,
               population=0,
               population_year=0,
               per_capita=0.0,
               conversion_rate=0.0,
               crop_type=None):
    data = pd.DataFrame()
    if commodity_data and population_data:
        commodity_df = create_commodity_data(commodity_data, crop_type)
        ltt = run_ltt(commodity_df, conversion_rate=conversion_rate, crop_type=crop_type)
        cagr = run_cagr(commodity_df, conversion_rate=conversion_rate, crop_type=crop_type)
        arima = run_arima(commodity_df, conversion_rate=conversion_rate)
        consumption = project_consumption(file=population_data, current_population=population, year=population_year,
                                          per_capita=per_capita)

        # combine the datasets into 1
        data = ltt.merge(cagr, on='year', how='left').merge(arima, on='year', how='left').merge(consumption, on='year', how='left')
        print(data.tail())

    else:
        raise Exception('Please upload commodity and population data')

    return data


def create_commodity_data(file=None, crop_type='crop'):
    commodity_df = pd.read_csv(file, thousands=',', header='infer')
    print(commodity_df.head())

    if crop_type.lower() == 'crop':
        if len(commodity_df.axes[1]) != 3:
            raise Exception(
                'File must have 3 columns, namely: year, production, area harvested. Your file has %s' % str(
                    len(commodity_df.axes[1])))
        else:
            commodity_df.columns = ['year', 'production', 'area_harvested']
            commodity_df['yield'] = commodity_df.production / commodity_df.area_harvested
    else:
        if len(commodity_df.axes[1]) != 2:
            raise Exception(
                'File must have 2 columns, namely: year and production. Your file has %s' % str(
                    len(commodity_df.axes[1])))
        else:
            commodity_df.columns = ['year', 'production']

    return commodity_df


def project_consumption(file=None, current_population=0, year=0, per_capita=0):
    df = pd.read_csv(file, thousands=',', header='infer')
    df.columns = ['year', 'rate']
    #     response = dict()

    projection = pd.DataFrame()

    if int(df.year.min()) - year != 1:
        raise Exception('Sorry, the population growth rate data must start with %s' % str(year + 1) +
                              'Your data starts with ' + str(df.year.min()) +
                              '. You can either change the baseline year or update the data frame to start with %s' % str(
                    year + 1))
    else:
        projection = pd.DataFrame(columns=['year', 'rate', 'population'])
        projection.year = df.year
        projection.rate = df.rate

        projected_population = []
        for i in range(len(df)):
            current_population = int(round(current_population * (1 + df.rate.iloc[i] / 100), 0))
            projected_population.append(current_population)
        projection['population'] = pd.Series(projected_population)
        projection['consumption'] = projection.population * per_capita / 1000  # divide by 1000 to get MT value

    #         response['data'] = projection.to_dict(orient='records')

    # multiply by per capita * 1000 to get MT value
    return projection[['year', 'consumption']]


if __name__ == '__main__':
    app.debug = True
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
