import requests
import pandas as pd
import numpy as np
import re
import os
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from time import sleep

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

user="FabianL"
password="190893"
login_dict = {'user': user,
              'pwd': password}


def get_html(N_start, N_end):
    html_dict = {}
    with requests.session() as s:
        login_url = "https://tippspiel.altenbernd.eu/login.php"
        res = s.post(login_url, data=login_dict)
        print("Downloading data")
        for N in tqdm(range(N_start, N_end)):
            url = f'https://tippspiel.altenbernd.eu/matchday.php?matchday_id={N}&season_id='
            html = s.get(url).content
            html_dict[str(N)] = html
            sleep(0.5)
    return html_dict

def get_df(html):
    df_list = pd.read_html(html, encoding='utf-8')
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
    html_dict = get_html(N_start, N_end)
    for N in range(N_start, N_end):
        html = html_dict[str(N)]
        spieltag = get_spieltag(html, N)
        season = get_season(html)
        if spieltag and season:
            print(f"Processing data for N={N}, Spieltag={spieltag}, Season={season}")
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

def store_data(N_start, N_end):
    if not os.path.exists("Bundesliga_data.csv"):
        my_df = download_data(N_start, N_end)
        my_df.to_csv("Bundesliga_data.csv")
    else:
        print("File Bundesliga_data.csv already exists")

def load_stored_data():
    df = pd.read_csv("Bundesliga_data.csv", index_col=[0], decimal=',', encoding='utf-8')
    return df

if __name__ == "__main__":
#    import ipdb; ipdb.set_trace()
#    store_data(1,250)
    df = load_stored_data()
    import ipdb; ipdb.set_trace()
    
#import ipdb; ipdb.set_trace()    
