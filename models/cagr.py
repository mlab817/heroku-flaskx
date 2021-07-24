import pandas as pd


from matplotlib import pyplot as plt
from models.mape import mape


def cagr(df=pd.DataFrame, column_name="", start_year=None, end_year=None, ahead=32):
    """
    Function to compute compounded annual growth rate and forecast

    :param df: pandas DataFrame
    :param column_name: column that will be computed (variable)
    :param start_year: Beginning of data
    :param end_year: Ending of data
    :param ahead: No. of years to be forecasted
    :return: results of CAGR forecasting
    """

    # assign the input dataframe to a local variable to avoid modifying data
    data = df

    # get first and last year of data to be used
    first = start_year if start_year else data['year'].iloc[0]
    last = end_year if end_year else data['year'].iloc[-1]

    first_index = data[data.year == first].index.values[0]
    last_index = data[data.year == last].index.values[0]
    print("first_index: ", first_index, "last_index: ", last_index)
    # create a copy of the dataframe between the first and last years
    df2 = data.loc[first_index:last_index]

    # get the first and last values of the variable
    start_value = df2[column_name].iloc[0]
    end_value = df2[column_name].iloc[-1]

    # note: CAGR returned is its absolute value
    cagr_value = (end_value / start_value) ** (1 / len(df2)) - 1
    # print("cagr of {columnName} is {cagr}%".format(columnName=column_name, cagr=cagr_value * 100))

    # create a new dataframe to compare the prediction and actual values
    rows = [[first + i, start_value * (1 + cagr_value) ** i, df2[column_name].iloc[i] if i < len(df2) else None] for i in
            range(len(df2) + ahead)]
    prediction = pd.DataFrame(rows, columns=["year", "predicted", "actual"])
    # prediction.set_index('year')

    # forecast data
    mape_value = mape(prediction.actual, prediction.predicted)

    # plot the predicted vs actual values
    # plt.plot(prediction.year, prediction.predicted)
    # plt.plot(prediction.year, prediction.actual)
    # plt.show()

    # let's create a response json data and populate with data
    response_json = dict()
    response_json['data'] = prediction.to_dict(orient="records")
    response_json['cagr'] = cagr_value
    response_json['mape'] = mape_value

    # create JSON data for consumption of the web application
    return response_json
