import numpy as np


def mape(y_actual, y_predicted):
    """
    Function to get value of mean absolute percentage error
    see https://www.statisticshowto.com/mean-absolute-percentage-error-mape/

    :param y_actual:
    :param y_predicted:
    :return: MAPE
    """

    mean_absolute_percentage_error = np.mean(np.abs((y_actual - y_predicted) / y_actual))
    # print('MAPE absolute value is: ', mean_absolute_percentage_error)
    return mean_absolute_percentage_error
