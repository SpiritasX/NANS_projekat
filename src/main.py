import sys
import pandas as pd
import loader
import map

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("Ignorisi ovaj tekst, dodacu ove funkcionalnosti jednog dana...")
        print("To analyse a specific station, run \"python src/main \"Kikinda Centar\"\" or similar")
        print("To see the list of all available stations, run \"python src/main stations\"")

    df = loader.load_all_tables(plot=True)

    m = map.map_of_serbia()
    stations = pd.read_csv('..\\data\\stanice.csv')
    map.draw_stations(m, stations['Longitude'], stations['Latitude'])