import sys
import pandas as pd
import map
import time_series
import utils
import clustering
import linear_regression
import pickle


def find_arg(args, arg):
    for i in range(len(args)):
        if args[i] == arg:
            return args[i + 1]
    return None


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print(f"Run \"{sys.argv[0]} --help\"")
        sys.exit(0)
    elif args[0] == "--help":
        print(f"Usage \"{sys.argv[0]} [options]\"")
        print("Options:")
        print("\t--time-series\t\t\tDisplay time series analysis (Use with --station)")
        print("\t--clustering\t\t\tDisplay clustering analysis")
        print("\t--linear-regression\t\tDisplay linear regression analysis")
        print("\t--save-to-file <file>\t\tSave the kmeans models to a file. (Use with --clustering)")
        print("\t--load-from-file <file>\t\tLoad the kmeans models from a file. (Use with --clustering) (DO NOT RUN)")
        print("\t--stations\t\t\tDisplay a list of all stations in Serbia")
        print("\t--station <name>\t\tStation for which to do the analysis (\"Kikinda Centar\")")
        print("\t--years <string>\t\tA string list of years to analyse (\"2018 2019 2020\")")
        print("\t--polutant <name>\t\tWhich polutant to analyse (\"co\", \"no2\", \"o3\", \"pm2.5\", \"pm10\", \"so2\")")
        sys.exit(0)
        # print("To analyse a specific station, run \"python src/main \"Kikinda Centar\"\" or similar")
        # print("To see the list of all available stations, run \"python src/main stations\"")

    station = find_arg(args, "--station")
    polutant = find_arg(args, "--polutant")
    years = find_arg(args, "--years")
    if years is not None:
        years = list(years.split())

    df = utils.load_all_tables(plot=True, years=years, file=polutant)

    if "--print-all-stations" in args:
        df = utils.load_stations()
        for index, station in df.iterrows():
            if station["Pripada mrezi"] == "SEPA":
                print(station["Naziv stanice"])

    if "--clustering" in args and years is not None:
        if "--save-to-file" in args:
            file_name = find_arg(args, "--save-to-file")

            # spajamo tabele za podacima
            pollution_data = utils.load_all_tables(years=years, file='pm2.5')
            # popunjavano na vrednosti
            stations = pd.read_csv('data\\stanice.csv', header=None)
            for stanica in pollution_data:
                pollution_data = utils.fillna_mean(pollution_data, stanica, chunk=80)

            data_scaled = utils.normalize(pollution_data)
            clustering.elbowMethod(data_scaled)

            stations_info = utils.transposing(pd.read_csv('data\\stanice.csv', header=None))
            weather_data = {}
            for station in pollution_data.columns:
                weather_data[station] = pd.read_csv("data\\LinReg\\" + station + ".csv").set_index('datetime')

            result = []
            for index in pollution_data.index:
                day_data = []
                for station in pollution_data.columns:
                    station_data = [
                        float(stations_info[station]['Latitude']),
                        float(stations_info[station]['Longitude']),
                        float(stations_info[station]['Nadmorska visina'][:-1]),
                        float(pollution_data[station][index]),
                        float(weather_data[station]['temp'][str(index)[:10]]),
                        float(weather_data[station]['humidity'][str(index)[:10]]),
                        float(weather_data[station]['windspeed'][str(index)[:10]]),
                        float(weather_data[station]['sealevelpressure'][str(index)[:10]])
                    ]
                    day_data.append(station_data)
                result.append(day_data)

            # odredjujemo i cuvamo klastere
            data_to_save = clustering.save_clusters(result)

            with open("data\\" + file_name + ".pkl", "wb") as fp:
                pickle.dump(data_to_save, fp)
    elif "--clustering" in args and "--load-from-file" in args:
        # Oblik ovih podataka je lista gde svaki element predstavlja jedan
        # fittovan model koji odgovara jednom danu
        # Pristupanje klasterima jednog dana bi se vrsilo sa clusters[i].labels_
        file_name = find_arg(args, "--load-from-file")
        with open("data\\" + file_name + ".pkl", "rb") as fp:
            data = pickle.load(fp)
            num_of_frames_from_end = int(input("Number of days to animate (up to 732): "))
            clustering.clusters_to_video(data, num_of_frames_from_end)

    if "--linear-regression" in args and station is not None:
        df = linear_regression.making_table(station)
        linear_regression.linear_regression(df)

    if "--time-series" in args and station is not None:
        df_filled = utils.fillna_mean(df, station, 64)
        time_series.time_series_trend(df_filled, station)
        time_series.yearly(df_filled, station)
        time_series.weekly(df_filled, station)

        model = time_series.PFM(df, station)
        time_series.plot_PFM(model, 365)
