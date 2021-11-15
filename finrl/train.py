import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from sklearn import preprocessing
import datetime
from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.env_stock_trading.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.neo_finrl.env_stock_trading.env_stocktrading_stoploss import StockTradingEnvStopLoss
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, tear_plot
import itertools

matplotlib.use("Agg")


def train_stock_trading():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_PLUS,
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

    # Training & Trading data split
    train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
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
        # "stoploss_penalty": 0.9,
        # "profit_loss_ratio": 1.2,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": np.array([4, 0, 3]),
        "print_verbosity": 5,
        # "discrete_actions": True,
        # "patient": False,
        # "print_verbosity": 100,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs, random_start=True)
    env_train, _ = e_train_gym.get_sb_env()
    # env_train, _ = e_train_gym.get_multiproc_env(n=6)
    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("sac", policy_kwargs=dict(net_arch=[64, 64],
                                                          activation_fn=nn.ReLU), verbose=1)
    trained_sac = agent.train_model(
        model=model_sac, tb_log_name="sac", total_timesteps=50000)
    trained_sac.save("./" + config.TRAINED_MODEL_DIR + "/" + "trained_sac_" + now)

    # model_ppo = agent.get_model("ppo", policy_kwargs=dict(net_arch=[dict(pi=[512, 512], vf=[128, 128])],
    #                                                       activation_fn=nn.ReLU), verbose=10000)
    # trained_ppo = agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=100000)
    # trained_ppo.save("./" + config.TRAINED_MODEL_DIR + "/" + "trained_ppo" + now)

    print("==============Start Trading===========")
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
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
    # # S&P 500: ^GSPC
    # # Dow Jones Index: ^DJI
    # # NASDAQ 100: ^NDX
    # fig = tear_plot(df_account_value, value_col_name="total_assets", baseline_ticker="^DJI")
    fig = tear_plot(df_account_value, baseline_ticker="^DJI")
# def train_portfolio_allocation():
