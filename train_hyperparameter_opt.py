import random

import pandas as pd
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
from IPython.core.display import clear_output
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime
import optuna
from torch import nn
from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.env_stock_trading.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.neo_finrl.env_stock_trading.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, tear_plot
import itertools


class LoggingCallback:
    def __init__(self, threshold, trial_number, patience):
        '''
        threshold:int tolerance for increase in sharpe ratio
        trial_number: int Prune after minimum number of trials
        patience: int patience for the threshold
        '''
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = []  # Trials list for which threshold is reached

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        # Setting the best value in the current trial
        study.set_user_attr("previous_best_value", study.best_value)
        # study.set_user_attr("previous_best_value", study.)

        # Checking if the minimum number of trials have pass
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # Checking if the previous and current objective values have the same sign
            if previous_best_value * study.best_value >= 0:
                # Checking for the threshold condition
                if abs(previous_best_value - study.best_value) > self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    # If threshold is achieved for the patience amount of time
                    if len(self.cb_list) > self.patience:
                        print('The study stops now...')
                        print('With number', frozen_trial.number, 'and value ', frozen_trial.value)
                        print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                        study.stop()


# Calculate the daily return
def calculate_return(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    return df['daily_return'].mean()


# Calculate the Sharpe ratio
def calculate_sharpe(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    # For env_stocktrading_cashpenalty
    # df['daily_return'] = df['total_assets'].pct_change(1)
    if df['daily_return'].std() != 0:
        sharpe = (252 ** 0.5) * df['daily_return'].mean() / \
                 df['daily_return'].std()
        return sharpe
    else:
        return 0


def calculate_std(df):
    df['daily_return'] = df['account_value'].pct_change(1)
    # For env_stocktrading_cashpenalty
    # df['daily_return'] = df['total_assets'].pct_change(1)
    if df['daily_return'].std() != 0:
        std = df['daily_return'].std()
        return std
    else:
        return 0


def sample_ppo_params(trial: optuna.Trial):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    # n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256])
    # gamma = trial.suggest_float("gamma", 0.85, 0.99)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.0008)
    # ent_coef = trial.suggest_discrete_uniform("ent_coef", 0.01, 0.2, 0.01)
    # clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    # n_epochs = trial.suggest_int("n_epochs", 1, 10, 1)
    # gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # net_arch = trial.suggest_categorical("net_arch", ["m", "h", "Rm", "Rh"])
    # net_arch = trial.suggest_categorical("net_arch", ["s", "m", "h"])
    # p_net = trial.suggest_categorical("p_net", [64, 128, 256, 512])
    # v_net = trial.suggest_categorical("v_net", [64, 128, 256, 512])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu"])
    buffer_size  = trial.suggest_int("buffer_size", 70000, 300000, 10000)
    asset_re  = trial.suggest_int("asset_re", 0, 5, 0.5)
    return_re = trial.suggest_int("return_re", 0, 5, 0.5)
    shapre_re = trial.suggest_int("shapre_re", 0, 5, 0.5)
    reward_scale = np.array([asset_re, return_re, shapre_re])
    tech_index = trial.suggest_categorical("tech_index", ["basic", "b_boll", "adv"])
    # if batch_size > n_steps*8:
    #     batch_size = n_steps*8

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # reward_scale = trial.suggest_discrete_uniform("reward_scale", 0.00, 5, 0.01)

    tech_index = {
        "basic": ["macd", "rsi_30", "cci_30", "dx_30"],
        "b_boll": ["macd", "rsi_30", "boll_ub", "boll_lb"],
        "adv":["macd", "boll_ub", "boll_lb", "rsi_30", "cci_14", "cci_30", "dx_30", "close_30_sma", "close_60_sma"],
    }[tech_index]

    # Independent networks usually work best
    # when not working with images
    # net_arch = {
    #     "xs": [64, 64],
    #     "s": [128, 128],
    #     "m": [256, 256],
    #     "h": [512, 512],
    #     # "3s": [128, 128, 128],
    #     # "3m": [256, 256, 256],
    #     # "3h": [512, 512, 512],
    #     # "Rxs": [64, 32],
    #     "Rs": [128, 64],
    #     "Rm": [256, 128],
    #     "Rh": [512, 256],
    #     # "small": [dict(pi=[64, 64], vf=[64, 64])],
    #     # "medium": [dict(pi=[256, 256], vf=[256, 256])],
    #     # "s": [dict(pi=[128, 128], vf=[128, 128])],
    #     # "m": [dict(pi=[256, 256], vf=[128, 128])],
    #     # "h": [dict(pi=[512, 512], vf=[128, 128])],
    # }[net_arch]

    # net_arch = [dict(pi=[p_net, p_net], vf=[v_net, v_net])]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    model_params = {
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "gamma": 0.9,
                    "learning_rate": learning_rate,
                    "ent_coef": "auto_0.2",
                    # "ent_coef": ent_coef,
                    # "n_steps": n_steps,
                    # "clip_range": clip_range,
                    # "n_epochs": n_epochs,
                    # "gae_lambda": gae_lambda,
                    # "max_grad_norm": max_grad_norm,
                    # "vf_coef": vf_coef,
                    # "sde_sample_freq": sde_sample_freq,
                    }
    # policy_params = dict(net_arch=net_arch, activation_fn=activation_fn)
    # policy_params = dict(net_arch=net_arch)
    # policy_params = dict(net_arch=[512, 512], activation_fn=activation_fn)
    return tech_index, reward_scale, model_params


# This is our objective for tuning
def objective(trial: optuna.Trial):
    # Trial will suggest a set of hyperparamters from the specified range
    tech_index, reward_scale, model_params= sample_ppo_params(trial)

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_index,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)

    # Training & Trading data split
    train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
            1
            + 2 * stock_dimension
            + len(tech_index) * stock_dimension
    )


    env_kwargs = {
        "hmax": 100,
        "initial_amount": 20000,
        "buy_cost_pct": 0.01,
        "sell_cost_pct": 0.01,
        # "stoploss_penalty": 0.85,
        # "profit_loss_ratio": 1.2,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_index,
        # "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": reward_scale,
        "print_verbosity": 20,
        # "shares_increment": 100,
        # "discrete_actions": True,
        # "patient": False,
        # "print_verbosity": 10000,
    }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs, random_start=True)
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=None, **env_kwargs, )
    env_train, _ = e_train_gym.get_sb_env()
    # env_train, _ = e_train_gym.get_multiproc_env(n=6)
    print(type(env_train))
    agent = DRLAgent(env=env_train)
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    model_sac = agent.get_model("sac", model_kwargs=model_params, verbose=0)

    # model_ppo = agent.get_model("ppo", model_kwargs=model_params, policy_kwargs=policy_params, verbose=100)
    # You can increase it for better comparison
    # trained_ppo = agent.train_model(model=model_ppo,
    #                                 tb_log_name='ppo',
    #                                 total_timesteps=20000,
    #                                 )

    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=40000,
                                    )
    # clear_output(wait=True)
    # For the given hyperparamters, determine the account value in the trading period
    # df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,
    #                                                        environment=e_trade_gym
    #                                                        )

    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_sac,
                                                           environment=e_trade_gym
                                                           )
    baseline_df = get_baseline(
        ticker='^DJI', start=config.START_TRADE_DATE, end=config.END_DATE
    )
    baseline_df = pd.merge(df_account_value['date'], baseline_df, how='left', on='date')
    baseline_df = baseline_df.fillna(method='ffill').fillna(method='bfill')
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    perf_stats_all = backtest_stats(df_account_value, baseline_return=baseline_returns)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    alpha = perf_stats_all.loc['Alpha']

    return alpha


print("==============Start Fetching Data===========")
df = YahooDownloader(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    ticker_list=config.DOW_30_TICKER,
).fetch_data()

# Create a study object and specify the direction as 'maximize'
# As you want to maximize sharpe
# Pruner stops not promising iterations
# Use a pruner, else you will get error related to divergence of model
# You can also use Multivariate samplere
# sampler = optuna.samplers.TPESampler(multivariate=True)
sampler = optuna.samplers.RandomSampler()
# study = optuna.create_study(study_name="sac_study", direction='maximize',
#                             sampler=sampler, pruner=optuna.pruners.HyperbandPruner(),)
study = optuna.create_study(direction='maximize',
                            sampler=sampler, pruner=optuna.pruners.HyperbandPruner(),
                            storage="mysql+pymysql://optuna@localhost/optuna")
# study = optuna.multi_objective.create_study(study_name="ppo_study", directions=['maximize','maximize'],
#                             sampler=sampler)
# study = optuna.create_study(directions=['maximize','maximize'])

logging_callback = LoggingCallback(threshold=1e-4, patience=30, trial_number=5)
# You can increase the n_trials for a better search space scanning
study.optimize(objective, n_trials=50, catch=(ValueError,), callbacks=[logging_callback])
# study.optimize(objective, n_trials=50)
print('Best params: ' + str(study.best_params))
fig = optuna.visualization.plot_contour(study)
# fig = optuna.visualization.plot_pareto_front(study)
fig.show()
