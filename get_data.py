import requests
import pandas as pd
import numpy as np
import re
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_hmtl(N):
    url = f'https://tippspiel.altenbernd.eu/matchday.php?matchday_id={N}&season_id='
    html = requests.get(url).content
    return html

def get_df(html):
    df_list = pd.read_html(html)
    df = df_list[-1]
    return df
    
def get_spieltag(html, N):
    try:
        spieltag_str = re.findall(r"<option value=\\\'"+str(N)+r"\\\' selected.{0,50}</option>", str(html))[0]
        spieltag = re.findall("(\d+). Spieltag", spieltag_str)[0]
        return spieltag
    except:
        return
    
def get_season(html):
    try:
        season = re.findall("selected >Bundesliga (.{0,9})", str(html))[0]
        return season
    except:
        return

def download_data(N_start, N_end):
    my_df = pd.DataFrame(columns=["Spieltag", "Season", "Name", "Team1", "Team2", "Guess", "Result", "Points"])
    for N in range(N_start, N_end):
        html = get_hmtl(N)
        spieltag = get_spieltag(html, N)
        season = get_season(html)
        if spieltag and season:
            print(f"Downloading data for N={N}, Spieltag={spieltag}, Season={season}")
            df = get_df(html)
            # drop last row, somehow this row is repeated
            df.drop(df.tail(1).index,inplace=True)
        else:
            print(f"Invalid data for N={N}")
            continue
        
        cols = df.columns.drop(["Platz", "Punkte", "Name"])
        for row in df.to_dict(orient='index').values():
            for col in cols:
                team1, _, team2, result = col.split(" ")
                try:
                    guess, points = row[col].split(" ")
                except:
                    guess = np.NaN
                    points = 0
                my_dict = {"Spieltag": spieltag,
                           "Season": season,
                           "Name": row['Name'],
                           "Team1": team1,
                           "Team2": team2,
                           "Guess": guess,
                           "Result": result,
                           "Points": points
                           }
                my_df = my_df.append(my_dict, ignore_index=True)
    return my_df

def store_data():
    if not os.path.exists("Bundesliga_data.csv"):
        my_df = download_data(1, 250)
        my_df.to_csv("Bundesliga_data.csv")
    else:
        print("File Bundesliga_data.csv already exists")

def load_stored_data():
    df = pd.read_csv("Bundesliga_data.csv", index_col=[0])
    return df

if __name__ == "__main__":
#    import ipdb; ipdb.set_trace()
    store_data()
    df = load_stored_data()
    import ipdb; ipdb.set_trace()
    
#import ipdb; ipdb.set_trace()    
