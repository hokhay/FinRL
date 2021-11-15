import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
# matplotlib.use("TkAgg")
import datetime

from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.env_stock_trading.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, trx_plot, tear_plot, get_baseline, get_daily_return

import itertools

print("==============Start Fetching Data===========")
df = YahooDownloader(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    ticker_list=config.USER_DEFINED,
).fetch_data()
print("==============Start Feature Engineering===========")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
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

# Trading & training data split
train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

# calculate state action space
stock_dimension = len(trade.tic.unique())
state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
)

env_kwargs = {
    "hmax": 100,
    "initial_amount": 20000,
    "buy_cost_pct": 0.01,
    "sell_cost_pct": 0.01,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 2,
    "initial": True,
    "previous_state": [3500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 15, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0]
}

e_train_gym = StockTradingEnv(df=trade, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
agent = DRLAgent(env=env_train)
trained_sac = agent.get_model("sac")

trained_model = trained_sac.load("./" + config.TRAINED_MODEL_DIR + "/" + "trained_sac_20211107-16h28_best")

e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_model, environment=e_trade_gym
)
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

df_account_value.to_csv(
    "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
)

df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

print("==============Get Backtest Results===========")
# get baseline return
baseline_df = get_baseline(
        ticker='^DJI', start=config.START_TRADE_DATE, end=config.END_DATE
    )
baseline_df = pd.merge(df_account_value['date'], baseline_df, how='left', on='date')
baseline_df = baseline_df.fillna(method='ffill').fillna(method='bfill')
baseline_returns = get_daily_return(baseline_df, value_col_name="close")
perf_stats_all = backtest_stats(df_account_value, baseline_return=baseline_returns)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

print("==============Compare to DJIA===========")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
# fig = tear_plot(df_account_value, value_col_name="total_assets", baseline_ticker="^DJI")
fig = tear_plot(df_account_value, baseline_ticker="^DJI")
