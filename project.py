import numpy as np
import pandas as pd
import requests
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime as dt

URL_POPULATION = "https://restcountries.eu/rest/v2/name/{}"
URL_COVID = "https://corona.lmao.ninja/v2/historical/{}"


def func(x, a, b, c):
    return a*np.exp(b*x)+c


def load_json(country: str, url: str):
    request = requests.get(url.format(country))
    if request.status_code != 200:
        print("Status code: {}, not OK".format(request.status_code))
        sys.exit()
    return request.json()


def get_population(country):
    data = load_json(country, URL_POPULATION)
    return data[0]["population"]


def get_cases_on_date(country, date):
    population = get_population(country)
    # print(population)
    data = load_json(country, URL_COVID)
    data = data["timeline"]["cases"][date]
    return data, data/population*1e6


def get_data(country):
    data = load_json(country, URL_COVID)
    population = get_population(country)
    ds = []
    y = []
    y_abs = []
    for i, xd in enumerate(data["timeline"]["cases"]):
        xdStr = "{}".format(xd)
        num = data["timeline"]["cases"][xdStr]
        ds.append(xd)
        y.append(num/population*1e6)
        y_abs.append(num)

    ds = pd.to_datetime(ds)
    df = pd.DataFrame({
        "ds": ds,
        "y": y,
        "y_abs": y_abs
    })
    df = df[df["y_abs"] > 0]
    return df


def get_xticks(dates):
    return dates.apply(lambda x: x.strftime("%d/%m"))


def find_coeffs(x, y):
    xd = curve_fit(func, x, y, p0=(1, 0.1, 1))
    return xd[0][0], xd[0][1], xd[0][2]


def test_func(dates, delta):
    dates_future = dates
    for i in range(1, delta):
        to_add = dates.tail(1)+dt.timedelta(i)
        to_add.index += i
        dates_future = pd.concat([dates_future, to_add])
        # print(dates.tail(1)+dt.timedelta(i))
    return dates_future


def predict_data(x, y, delta):
    x_predict = range(x[0], x[len(x)-1]+delta)
    A, B, C = find_coeffs(x, y)
    #print(A, B, C)
    return x_predict, func(x_predict, A, B, C)


def get_data_predict(df, delta, country):
    population = get_population(country)
    x = df.index
    y = df["y"]
    y_abs = df["y_abs"]
    x_predict = range(x[0], x[len(x)-1]+delta)
    #A, B, C = find_coeffs(x, y)
    A2, B2, C2 = find_coeffs(x, y_abs)
    y_predict = []
    y_abs_predict = func(x_predict, A2, B2, C2)
    for x in y_abs_predict:
        y_predict.append(x/population*1e6)
    df_predict = pd.DataFrame({
        "ds": test_func(df["ds"], delta),
        "y": y_predict,
        "y_abs": y_abs_predict
    })
    return df_predict


if __name__ == '__main__':
    today = (dt.datetime.today()-dt.timedelta(2)).strftime("%#m/%#d/%y")
    country = "poland"
    df_poland = get_data(country)
    delta = 7
    df_poland_predict = get_data_predict(df_poland, delta, country)
    print(df_poland_predict)
    plt.figure(0)
    plt.plot(df_poland.index, df_poland["y_abs"],
             color="orange", label="number of cases")
    # plt.plot(df_poland.index, df_poland["y"],
    #         color="blue", label="measured", linestyle=":")
    plt.xticks(df_poland.index, get_xticks(
        df_poland["ds"]), rotation="vertical")
    plt.title("Koronawirus w Polsce")
    # plt.axvline(df_poland[df_poland["ds"] ==
    #                      "24/3/2020"].index, color="red", label="xd")
    plt.legend(loc="upper left")

    print(get_cases_on_date(country, today))
    print(today)

    plt.figure(1)
    country2 = "china"
    df_china = get_data(country2)
    plt.plot(df_china["ds"], df_china["y_abs"],
             color="orange", label="number of cases")
    plt.xticks(rotation="45")
    plt.legend(loc="upper left")
    plt.title("Coronavirus in China")

    country3 = "italy"
    df_italy = get_data(country3)
    df_italy_predict = get_data_predict(df_italy, delta, country3)

    country4 = "usa"
    df_usa = get_data(country4)
    df_usa_predict = get_data_predict(df_usa, delta, country4)
    # asdasda
    # plt.figure(1)
    # country2 = "usa"
    # df_usa = get_data(country2)
    # df_usa_predict = get_data_predict(df_usa, delta, country2)
    # plt.plot(df_usa_predict.index,
    #          df_usa_predict["y_abs"], color="orange", label="predicted usa")
    # plt.plot(df_poland_predict.index,
    #          df_poland_predict["y_abs"], color="blue", label="predicted poland")
    # plt.title("usa vs italy absolute")

    # plt.figure(2)
    # country3 = "italy"
    # plt.plot(df_usa.index, df_usa["y"], color="orange", label="usa")
    # plt.plot(df_poland.index, df_poland["y"], color="blue", label="italy")
    # plt.title("usa vs italy per million")
    # plt.legend(loc="upper left")
    # plt.xticks(df_usa.index, get_xticks(df_usa["ds"]), rotation="45")
    plt.show()
