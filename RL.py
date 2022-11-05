import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import pandas as pd
import random
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from get_data import tendency, load_stored_data, add_guess_df, calc_points_df, result_str_to_ints, create_state_generator, get_prev_n_results

torch.set_num_threads(4)

class TippSpielEnv(Env):
    def __init__(self, n):
        self.df = load_stored_data
        self.state_dict = {}
        # How many previous results do we want to include in observation space
        self.n = n
        self.max_goals = 9
        # guess in form of two integers in range [0,self.max_goals-1]
        self.action_space = MultiDiscrete([self.max_goals, self.max_goals]) #Box(0,10, shape=(1,))
        # Array of n previous guesses for both teams -> 4n numbers
        self.observation_space = MultiDiscrete(n*4*[self.max_goals])
        self.state = np.random.randint(0,self.max_goals, size=4*n)
        
    def step(self, action):
        action_guess = str(action[0]) + ":" + str(action[1])
        df_new_guess = add_guess_df(self.state_dict["df"], action_guess)
        df_new_guess = calc_points_df(df_new_guess, check=False)
        points_for_guess = df_new_guess[df_new_guess.Name == "New guess"]["calc_points"].values[0]

        # reward corresponds to how many points the new guess resulted in
        reward = points_for_guess

        try:
            self.state_dict = next(self.state_gen)
            self.set_state()
            done = False
        except StopIteration:
            done = True
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def set_state(self):
        results_as_list = list(self.state_dict.values())
        results = results_as_list[0] + results_as_list[1]
        results = [digit for res in results for digit in result_str_to_ints(res)]
        assert len(results) == 4*self.n
        self.state = results
    
    def reset(self):
        self.state_gen = create_state_generator(self.n)
        self.state_dict = next(self.state_gen)
        self.set_state()
        return self.state

n_prev = 5
env=TippSpielEnv(n_prev)

log_path = os.path.join('Training', 'Logs')
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
#model.learn(total_timesteps=20)
#model.save('PPO')
model = PPO.load("PPO", env)
#for i in range(1):
#    model.learn(total_timesteps=200000)
#model.save('PPO')
#eval_res = evaluate_policy(model, env, n_eval_episodes=10)

def predict_spieltag(model, spieltag, season):
    df = load_stored_data()
    df = df.loc[(df.Spieltag == spieltag) & (df.Season == season)]
#    import ipdb; ipdb.set_trace()
    teams1 = df["Team1"].unique().tolist()
    teams2 = df["Team2"].unique().tolist()
    final_df = pd.DataFrame()
    for t1, t2 in zip(teams1, teams2):
#        import ipdb; ipdb.set_trace()
        prev_n_results_1 = get_prev_n_results(spieltag, season, t1, n_prev)
        prev_n_results_2 = get_prev_n_results(spieltag, season, t2, n_prev)
        results = prev_n_results_1 + prev_n_results_2
        results = [digit for res in results for digit in result_str_to_ints(res)]
        action_guess, _ = model.predict(results, deterministic=True)
        action_guess_str = str(action_guess[0]) + ":" + str(action_guess[1])
        df_new_guess = add_guess_df(df.loc[df.Team1 == t1], action_guess_str)
        df_new_guess = calc_points_df(df_new_guess, check=False)
        final_df = final_df.append(df_new_guess)
    spieltag_result = final_df.groupby("Name")["calc_points"].sum().sort_values(ascending=False)
    print(spieltag_result)
    import ipdb; ipdb.set_trace()
    print("")
        

    
predict_spieltag(model, 14, "2021/2022")
#def test_predict():
#    sample_obs = env.observation_space.sample()
#    sample_obs = np.array([38 + np.random.randint(-3,3, size=10)]).astype(float)
#    sample_obs = [0,1,2,3,4,5,6,7,8,9]
#    action, _ = model.predict(sample_obs, deterministic=True)
#    print(f"obs = {sample_obs}")
#    print(f"action = {action}")
#test_predict()
import ipdb; ipdb.set_trace()
