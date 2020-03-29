import numpy as np
import pandas as pd
import requests
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import datetime as dt

URL_POPULATION = "https://restcountries.eu/rest/v2/name/{}"
URL_COVID = "https://corona.lmao.ninja/v2/historical/{}"

# funkcja wykladnicza
def func(x, a, b, c):
    return a*np.exp(b*x)+c

# funkcja sluzaca do pobrania danych z API
def load_json(country: str, url: str):
    request = requests.get(url.format(country))
    if request.status_code != 200:
        print("Status code: {}, not OK".format(request.status_code))
        sys.exit()
    return request.json()

# funkcja sluzaca do pobrania liczby ludzi w danym kraju
def get_population(country):
    data = load_json(country, URL_POPULATION)
    return data[0]["population"]


# pobranie danych oraz wrzucenie ich w dataframea
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
    #df = df[df["y_abs"] > 0]
    return df

# get_xticks - funkcje pomocnicze do stworzenia ladniejszej osi OX 
def get_xticks(dates):
    return dates.apply(lambda x: x.strftime("%d/%m"))


def get_xticks_v2(dates):
    new_dates = []
    for i, date in enumerate(dates):
        if(i % 7 == 0):
            new_dates.append(date.strftime("%d/%m"))
        else:
            new_dates.append(date.strftime(" "))
    return new_dates

# funkcja sluzaca do dopasowywania wspolczynnikow do danych
def find_coeffs(x, y):
    xd = curve_fit(func, x, y, p0=(1, 0.1, 1))
    return xd[0][0], xd[0][1], xd[0][2]

# funkcja zwracajca przyszle daty, uzywana do tworzenia dataframea
def get_dates_predict(dates, delta):
    dates_future = dates
    for i in range(1, delta):
        to_add = dates.tail(1)+dt.timedelta(i)
        to_add.index += i
        dates_future = pd.concat([dates_future, to_add])
    return dates_future


# funkcja ktora zwraca data framea z przewidywanymi wartosciami dla danego kraju
def get_data_predict(df, delta, country):
    population = get_population(country)
    x = df.index
    y = df["y"]
    y_abs = df["y_abs"]
    x_predict = range(x[0], x[len(x)-1]+delta)
    #A, B, C = find_coeffs(x, y)
    A2, B2, C2 = find_coeffs(x, y_abs)
    print("{}: A={}, B={}, C={}".format(country, A2, B2, C2))
    y_predict = []
    y_abs_predict = func(x_predict, A2, B2, C2)
    for x in y_abs_predict:
        y_predict.append(x/population*1e6)
    df_predict = pd.DataFrame({
        "ds": get_dates_predict(df["ds"], delta),
        "y": y_predict,
        "y_abs": y_abs_predict
    })
    return df_predict


if __name__ == '__main__':
    delta = 7
    today = (dt.datetime.today()-dt.timedelta(1)).strftime("%#m/%#d/%y")

    # wykresy do omowienia wloch
    country3 = "italy"
    df_italy = get_data(country3)
    df_italy_predict = get_data_predict(df_italy, delta, country3)
    plt.figure(2)
    lockdown_italy = "2020-03-09"
    plt.plot(df_italy.index, df_italy["y_abs"],
             color="orange", label="number of cases")
    plt.xticks(df_italy.index, get_xticks_v2(
        df_italy["ds"]), rotation="vertical")
    plt.title("Number of cases of Coronavirus in Italy")
    # print(df_italy)
    plt.axvline(df_italy[df_italy["ds"] == lockdown_italy].index,
                color="red", label="lockdown date")
    plt.legend(loc="upper left")

    # drugi wykres (przewidywania)
    plt.figure(3)
    plt.plot(df_italy_predict.index,
             df_italy_predict["y_abs"], color="orange", label="number of cases")
    plt.xticks(df_italy_predict.index, get_xticks_v2(
        df_italy_predict["ds"]), rotation="vertical")
    plt.title("Predicted number of cases of Coronavirus in Italy")
    plt.axvline(df_italy[df_italy["ds"] == today].index,
                color="yellow", label="today")
    plt.axvline(df_italy[df_italy["ds"] == lockdown_italy].index,
                color="red", label="lockdown date")
    plt.legend(loc="upper left")

    # wykresy do omowienia usa
    country4 = "usa"
    df_usa = get_data(country4)
    df_usa_predict = get_data_predict(df_usa, delta, country4)
    plt.figure(4)
    plt.plot(df_usa.index, df_usa["y_abs"],
             color="orange", label="number of cases")
    plt.xticks(df_usa.index, get_xticks_v2(df_usa["ds"]), rotation="vertical")
    plt.title("Number of cases of Coronavirus in USA")
    plt.legend(loc="upper left")

    # drugi wykres (przewidywania)
    plt.figure(5)
    plt.plot(df_usa_predict.index,
             df_usa_predict["y_abs"], color="orange", label="number of cases")
    plt.xticks(df_usa_predict.index, get_xticks_v2(
        df_usa_predict["ds"]), rotation="vertical")
    plt.title("Predicted number of cases of Coronavirus in USA")
    plt.axvline(df_usa_predict[df_usa_predict["ds"]
                               == today].index, color="yellow", label="today")
    plt.legend(loc="upper left")

    # wykresy do porownania dalekiej przyszlosci (pokazanie przeciecia sie)
    df_italy_far = get_data_predict(df_italy, 14, "italy")
    df_usa_far = get_data_predict(df_usa, 14, "usa")
    plt.figure(7)
    plt.plot(df_italy_far.index,
             df_italy_far["y"], color="c", label="No of cases in Italy")
    plt.plot(df_usa_far.index, df_usa_far["y"],
             color="m", label="No of cases in USA")
    plt.xticks(df_usa_far.index, get_xticks_v2(
        df_usa_far["ds"]), rotation="vertical")
    plt.axvline(df_usa_far[df_usa_far["ds"] == today].index,
                color="yellow", label="today")
    plt.legend(loc="upper left")
    plt.title("Predicted number of COVID-19 cases in further future")

    # porownanie dotychczasowej sytuacji
    plt.figure(8)
    plt.title("Recorded number of COVID-19 cases")
    plt.xticks(df_usa.index, get_xticks_v2(df_usa["ds"]), rotation="vertical")
    plt.plot(df_usa.index, df_usa["y"], color="m", label="No of cases in USA")
    plt.plot(df_italy.index, df_italy["y"],
             color="c", label="No of cases in Italy")
    plt.legend(loc="upper left")

    # porownanie przewidywan
    plt.figure(9)
    plt.title("Predicted number of COVID-19 cases")
    plt.xticks(df_usa_predict.index, get_xticks_v2(
        df_usa_predict["ds"]), rotation="vertical")
    plt.plot(df_usa_predict.index,
             df_usa_predict["y"], color="m", label="No of cases in USA")
    plt.plot(df_italy_predict.index,
             df_italy_predict["y"], color="c", label="No of cases in Italy")
    plt.axvline(df_usa_predict[df_usa_predict["ds"]
                               == today].index, color="yellow", label="today")
    plt.legend(loc="upper left")

    plt.show()
