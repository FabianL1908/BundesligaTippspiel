import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from get_data import tendency, load_stored_data, add_guess_df, calc_points_df, result_str_to_ints, create_state_generator

class TippSpielEnv(Env):
    def __init__(self, n):
        self.df = load_stored_data
        self.state_dict = {}
        # How many previous results do we want to include in observation space
        self.n = n
        # guess in form of two integers in range [0,6]
        self.action_space = MultiDiscrete([7, 7]) #Box(0,10, shape=(1,))
        # Array of n previous guesses for both teams -> 4n numbers
        self.observation_space = MultiDiscrete(n*[7,7,7,7])
        self.state = np.random.randint(0,7, size=4*n)
        
    def step(self, action):
        action_guess = str(action[0]) + ":" + str(action[1])
        df_new_guess = add_guess_df(self.state_dict["df"], action_guess)
        df_new_guess = calc_points_df(df_new_guess, check=False)
        points_for_guess = df_new_guess[df_new_guess.Name == "New guess"]["calc_points"].values[0]

        # reward corresponds to how many points the new guess resulted in
        reward = points_for_guess

        self.state_dict = next(self.state_gen)
        # Check if shower is done
        if self.state_dict is not None:
            done = False
        else:
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

env=TippSpielEnv(5)

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)
model.save('PPO')
model = PPO.load("PPO", env)
#model.learn(total_timesteps=50000)
#model.save('PPO')
eval_res = evaluate_policy(model, env, n_eval_episodes=10)

#def test_predict():
#    sample_obs = env.observation_space.sample()
#    sample_obs = np.array([38 + np.random.randint(-3,3, size=10)]).astype(float)
#    sample_obs = [0,1,2,3,4,5,6,7,8,9]
#    action, _ = model.predict(sample_obs, deterministic=True)
#    print(f"obs = {sample_obs}")
#    print(f"action = {action}")
#test_predict()
import ipdb; ipdb.set_trace()
