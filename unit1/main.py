from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import gym
env_name = "CarRacing-v0"
# First, we create our environment called LunarLander-v2
env = gym.make(env_name)

# Then we reset this environment
observation = env.reset()

for _ in range(20):
    # Take a random action
    action = env.action_space.sample()
    print("Action taken:", action)

    # Do this action in the environment and get
    # next_state, reward, done and info
    observation, reward, done, info = env.step(action)

    # If the game is done (in our case we land, crashed or timeout)
    if done:
        # Reset the environment
        print("Environment is reset")
        observation = env.reset()


# We create our environment with gym.make("<name_of_the_environment>")
env = gym.make(env_name)
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation


print("\n _____ACTION SPACE_____ \n")
# print("Action Space Shape", env.action_space.n)
# print("Action Space Sample", env.action_space.sample()) # Take a random action


# Create the environment
env = make_vec_env(env_name, n_envs=16)


# TODO: Define a PPO MlpPolicy architecture
# We use MultiLayerPerceptron (MLPPolicy) because the input is a vector,
# if we had frames as input we would use CnnPolicy
model = PPO('MlpPolicy', env, verbose=1)


# Improving the agent
# We added some parameters to fasten the training
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    learning_rate=0.0009,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.008,
    vf_coef=0.5,
    verbose=1)
# TODO: Train it for 500,000 timesteps
model.learn(total_timesteps=1200000)
# TODO: Specify file name for model and save the model to file
model_name = "ppo-" + env_name
model.save(model_name)
# model.load(model_name)


# TODO: Evaluate the agent
# Create a new environment for evaluation
eval_env = gym.make(env_name)

# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Print the results
print(f'Mean Reward: {mean_reward} -- Std Reward: {std_reward}')






import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_sb3 import package_to_hub

# TODO: Define the name of the environment
env_id = env_name

# TODO: Evaluate the agent
# Create a new environment for evaluation
eval_env = gym.make(env_name)

# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Print the results
print(f'Mean Reward: {mean_reward} -- Std Reward: {std_reward}')

# TODO: Define the model architecture we used
model_architecture = "PPO"

## TODO: Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = 'antonioricciardi/' + env_name

## TODO: Define the commit message
commit_message = "First upload of a PPO Lunar Lander agent"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)

# Note: if after running the package_to_hub function and it gives an issue of rebasing, please run the following code
# cd <path_to_repo> && git add . && git commit -m "Add message" && git pull
# And don't forget to do a "git push" at the end to push the change to the hub.