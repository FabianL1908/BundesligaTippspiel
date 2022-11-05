import requests
import pandas as pd
import numpy as np
import functools
import re
import os
from requests.auth import HTTPBasicAuth
from functools import partial
from tqdm import tqdm
from time import sleep

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def display_rows(N):
    pd.set_option('display.max_rows', N)
    return

display_rows(10)

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

def update_data():
    #N=250 is Spieltag 9 in season 2022/2023
    import ipdb; ipdb.set_trace()
    df = load_stored_data()
    for n in range(250, 256):
        try:
            add_df = download_data(n, n+1)
            df = df.append(add_df, ignore_index=True)
        except:
            break
    df.to_csv("Bundesliga_data.csv")


def load_stored_data():
    df = pd.read_csv("Bundesliga_data.csv", index_col=[0], decimal=',', encoding='utf-8')
    return df

def get_spieltag_df(spieltag, season):
    df = load_stored_data()
    df = df.loc[(df["Season"] == season) &
                (df["Spieltag"] == spieltag)]
    return df

def get_matches_of_spieltag(spieltag, season):
#    df = load_stored_data()
    spieltag_df = get_spieltag_df(spieltag, season).reset_index()
    matches = []
    for i in range(0,9):
        team1, team2 = spieltag_df.iloc[i][["Team1", "Team2"]]
        matches.append((team1, team2))
    return matches

def get_matches_df_list(spieltag, season):
    df = get_spieltag_df(spieltag, season)
    matches = get_matches_of_spieltag(spieltag, season)
    df_list = []
    for match in matches:
        df_m = df.loc[df.Team1 == match[0]]
        df_list.append(df_m)
    return df_list

def tendency(result):
    try:
        goal1, goal2 = result.split(":")
        goal1 = int(goal1); goal2 = int(goal2)
        if goal1 > goal2:
            return "win1"
        elif goal1 == goal2:
            return "draw"
        else:
            return "win2"
    except:
        return

def get_diff_and_goal(guess, result):
    guess1, guess2 = guess.split(":")
    res1, res2 = result.split(":")
    guess1 = int(guess1); guess2 = int(guess2); res1 = int(res1); res2 = int(res2)
    delta_diff = np.abs(np.abs(guess1-guess2) - np.abs(res1-res2))
    delta_goal = np.abs(guess1 + guess2 - res1 - res2)
    return (delta_diff, delta_goal)

def calc_points_spieltag(spieltag, season):
    df_list = get_matches_df_list(spieltag, season)
    out_df_list = []
    for df in df_list:
        df = calc_points_df(df)
        out_df_list.append(df)

    return out_df_list

def calc_points_df(df, check=True):
    result = df["Result"].iloc[0]
    tendencies = df["Guess"].apply(tendency)
    winners = df[tendencies==tendency(result)]
    diffs = winners["Guess"].apply(partial(get_diff_and_goal, result=result)).sort_values()
    if len(diffs) != 0:
        numbers = np.array(diffs.value_counts().sort_index().tolist())
        quotients = 1/np.cumsum(numbers)
        y = np.sum(numbers)*quotients*numbers
        points = 9/np.sum(y)*y/numbers
        if np.abs(np.sum(y) < 0.1):
            import ipdb; ipdb.set_trace()
        final_points = np.round(1 + points, 2)
        point_list = []
        for n, f in zip(numbers, final_points):
            point_list += n*[f]
    else:
        point_list = []    
    diff_calc_points = pd.DataFrame(diffs)
    diff_calc_points["calc_points"] = point_list
    df = pd.concat([df, diff_calc_points["calc_points"]], axis=1)
    df["calc_points"] = df["calc_points"].fillna(0.00)
    if check:
        assert np.abs(df["calc_points"].apply(lambda x: x-1 if x > 0.0 else 0.0).sum() - 9.0) < 0.1
        assert df["Points"].equals(df["calc_points"])

    return df

def add_guess_spieltag(spieltag, season, name, team1, team2, guess):
    df = load_stored_data()
    df = df.loc[(df.Spieltag == spieltag) &
                (df.Season == season) &
                (df.Team1 == team1) &
                (df.Team2 == team2)]
    my_dict = {"Spieltag": spieltag,
               "Season": season,
               "Name": name,
               "Team1": team1,
               "Team2": team2,
               "Guess": guess,
               "Result": df.Result.tolist()[0],
               "Points": np.NaN}
    df = df.append(my_dict, ignore_index=True)
    return df

def add_guess_df(df, guess):
    new_entry = df.iloc[0].to_dict()
    new_entry["Name"] = "New guess"
    new_entry["Guess"] = guess
    new_entry["Points"] = np.NaN
    df = df.append(pd.DataFrame(new_entry, index=[0]), ignore_index=True)
    return df

@functools.lru_cache(maxsize=1)
def get_spieltag_season_list():
    df = load_stored_data()
    spieltage = df["Spieltag"].unique()
    seasons = df["Season"].unique()
    return spieltage, seasons

def get_prev_n_spieltag_season(spieltag, season, n):
    spieltags, seasons = get_spieltag_season_list()
    if spieltag > n:
        return [(i, season) for i in range(spieltag-1, spieltag-n-1, -1)]
    else:
        prev_season = str(seasons[np.where(seasons == season)[0]-1][0])
        pt1 = [(i, season) for i in range(spieltag-1, 0, -1)]
        pt2 = [(i, prev_season) for i in range(34, 34-(n-spieltag+1), -1)]
        return pt1 + pt2

from joblib import Memory
memory = Memory("cachedir")
@memory.cache
def get_prev_n_results(spieltag, season, team, n):
    df = load_stored_data()
    prev_spieltag_season = get_prev_n_spieltag_season(spieltag, season, n)
    results = []
#    import ipdb; ipdb.set_trace()
    for pss in prev_spieltag_season:
        try:
            pss_df = df.loc[(df.Spieltag == pss[0]) & (df.Season == pss[1]) &
                            ((df.Team1 == team) | (df.Team2 == team))]
            #Here I throw away if home or away win
            result = pss_df["Result"].tolist()[0]
            if pss_df.Team2.tolist()[0] == team:
                result = result[2] + ":" + result[0]
            results.append(result)
        except:
            # if team has not played in Buli in the last n matches just set results to 0:0
            results.append["0:0"]
    return results

def result_str_to_ints(result):
    return [int(i) for i in result.split(":")]

def create_state_generator(n):
    df = load_stored_data()
    spieltags, seasons = get_spieltag_season_list()
#    import ipdb; ipdb.set_trace()
#    for row in df.to_dict(orient='index').values():
    for index, row in df.loc[df.Name == "Stefan Turek"].iterrows():
        spieltag = row["Spieltag"]
        season = row["Season"]
        team1 = row["Team1"]
        team2 = row["Team2"]
        if spieltag > n and season >= seasons[0] and season < "2022/2023":
            match_dict = {}
            match_dict[team1] = get_prev_n_results(spieltag, season, team1, n)
            match_dict[team2] = get_prev_n_results(spieltag, season, team2, n)
            match_dict["Spieltag"] = spieltag
            match_dict["Season"] = season
            match_dict["df"] = df.loc[(df.Spieltag == spieltag) & (df.Season == season)&
                                      (df.Team1 == team1)]
            yield match_dict
            

if __name__ == "__main__":
#    import ipdb; ipdb.set_trace()
#    store_data(1,250)
#    df = load_stored_data()
#    df_list = calc_points_spieltag(9, "2022/2023")
#    df = get_matches_df_list(9, "2022/2023")[0]
    update_data()
    import ipdb; ipdb.set_trace()
#    df = add_guess_spieltag(9, "2022/2023", "New Guess", "TSG", "BRE", "1:2")
#    df = add_guess_df(df, "1:2")
    df_new = calc_points_df(df, check=False)
#    get_prev_n_spieltag_season(7, "2021/2022", 7)
    results = get_prev_n_results(7, "2021/2022", "FCB", 7)
    my_gen = create_state_generator(5)
    print(next(my_gen))
    import ipdb; ipdb.set_trace()
    
#import ipdb; ipdb.set_trace()    
