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
            df = utils.load_all_tables(years=years, file='pm2.5')
            # popunjavano na vrednosti
            stations = pd.read_csv('data\stanice.csv', header=None)
            for stanica in df:
                df = utils.fillna_mean(df, stanica, chunk=80)

            data_scaled = utils.normalize(df)
            clustering.elbowMethod(data_scaled)

            #spajanje df i stations tabela
            stanice_transposed = utils.transposing(stations)
            df_spojeno = utils.inner_join_tables([stanice_transposed, df])
            
            # odredjujemo i cuvamo klastere
            data_to_save = clustering.save_clusters(df_spojeno, df)

            with open("data\\" + file_name + ".pkl", "wb") as fp:
                pickle.dump(data_to_save, fp)
    elif "--clustering" in args and "--load-from-file" in args:
        # Oblik ovih podataka je lista gde svaki element predstavlja jedan
        # fittovan model koji odgovara jednom danu
        # Pristupanje klasterima jednog dana bi se vrsilo sa clusters[i].labels_
        file_name = find_arg(args, "--load-from-file")
        data = None
        with open("data\\" + file_name + ".pkl", "rb") as fp:
            data = pickle.load(fp)
            clustering.clusters_to_video(data)
        
        

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
