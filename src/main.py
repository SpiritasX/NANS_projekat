import sys
import loader
import map
import time_series
import utils


def find_arg(args, arg):
    for i in range(len(args)):
        if args[i] == arg:
            return args[i + 1]


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print(f"Run \"{sys.argv[0]} --help\"")
        sys.exit(0)
    elif args[0] == "--help":
        print(f"Usage \"{sys.argv[0]} [options]\"")
        print("Options:")
        print("\t--time-series\t\t\tDisplay time series analysis")
        print("\t--print-all-stations\t\tDisplay a list of all stations in Serbia")
        print("\t--station <name>\t\tStation for which to do the analysis (\"Kikinda Centar\")")
        print("\t--years <string>\tA string list of years (\"2018 2019 2020\")")
        print("\t--file <name>\t\t\tFiles for which particle to analyse (\"co\", \"no2\", \"o3\", \"pm2.5\", \"pm10\", \"so2\")")
        sys.exit(0)
        # print("To analyse a specific station, run \"python src/main \"Kikinda Centar\"\" or similar")
        # print("To see the list of all available stations, run \"python src/main stations\"")

    station = None
    file = None
    years = None

    if "--station" in args:
        station = find_arg(args, "--station")

    if "--years" in args:
        years = list(find_arg(args, "--years").split())

    if "--file" in args:
        file = find_arg(args, "--file")

    if "--print-all-stations" in args:
        df = loader.load_stations()
        for index, station in df.iterrows():
            if station["Pripada mrezi"] == "SEPA":
                print(station["Naziv stanice"])

    df = None
    if years is not None and file is None:
        df = loader.load_all_tables(plot=True, years=years)
    elif years is None and file is not None:
        df = loader.load_all_tables(plot=True, file=file)
    elif years is not None and file is not None:
        df = loader.load_all_tables(plot=True, years=years, file=file)
    else:
        df = loader.load_all_tables(plot=True)

    if "--time-series" in args and station is not None:
        model = time_series.PFM(df, station)
        time_series.plot_PFM(model, 365)
        df = utils.fillna_mean(df, station, 40)
        time_series.yearly(df, station)
        time_series.weekly(df, station)
        time_series.time_series_trend(df, station)