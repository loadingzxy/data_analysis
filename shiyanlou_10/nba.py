import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import random
import csv
import math


base_elo = 1600
team_elos = {}
team_status = {}
x = []
y = []

folder = 'nba'

def init_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['RK', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['RK', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['RK', 'G', 'MP'], axis=1)

    team_status1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_status1 = pd.merge(team_status1, new_Tstat, how='left', on='Team')
    return team_status1.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos
    except:
        team_elos[team] = base_elo
        return team_elos[team]

def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))

    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank <= 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k*(1-odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank

def build_dataSet(all_data):
    print("Build data set ...")
    x = []
    skip = 0
    for index, row in all_data.iterrows():
        Wteam = row['Wteam']
        Lteam = row['Lteam']

        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        if row['Wloc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        team1_feature = [team1_elo]
        team2_feature = [team2_elo]

        for key, value in team_status.loc[Wteam].iteritems():
            team1_feature.append(value)
        for key, value in team_status.loc[Lteam].iteritems():
            team2_feature.append(value)

        if random.random() > 0.5:
            x.append(team1_feature + team2_feature)
            y.append(0)
        else:
            x.append(team2_feature + team1_feature)
            y.append(1)
        if skip == 0:
            print(x)
            skip = 1
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(x), y

if __name__ == '__main__':
    Mstat = pd.read_csv(folder + '/miscell.csv')
    Ostat = pd.read_csv(folder + '/opponent_per_game.csv')
    Tstat = pd.read_csv(folder + '/team_per_game.csv')

    team_status = init_data(Mstat, Ostat, Tstat)

    ressult_data = pd.read_csv(folder + '')