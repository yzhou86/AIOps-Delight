import pandas as pd
from h2o.estimators import H2OExtendedIsolationForestEstimator
from pyod.models.iforest import IForest
from tabulate import tabulate

from expiringdict import ExpiringDict
import time
import h2o
import matplotlib.pyplot as plt

abnormal_detect_cache = ExpiringDict(max_len=10000, max_age_seconds=1800)

_MIN_CALLS_ = 5
_MIN_COST_ = 0

_EIF_THRESHOLD_ = 0.8

def iforest_detection(df):
    X = df[["call_counts", "cost_sum", "suspicion_avg", "no_answered"]].values
    model = IForest(contamination=0.005)
    df["abnormal"] = model.fit_predict(X) == 1
    return df


def iforest_detection_h2o(df):
    h2o.init()
    predictors = ["counts", "cost", "suspicion", "no_answered"]
    predictors = ["counts", "cost", "suspicion"]
    predictors = ["counts_norm", "cost_norm", "suspicion_norm"]
    predictors = ["counts_norm", "cost_norm", "risk_norm"]
    eif = H2OExtendedIsolationForestEstimator(model_id="eif.hex",
                                              ntrees=100,
                                              sample_size=256,
                                              extension_level=len(predictors) - 1)
    eif.train(x=predictors,
              training_frame=h2o.H2OFrame(df))
    eif_result = eif.predict(h2o.H2OFrame(df))
    anomaly_score = eif_result["anomaly_score"].as_data_frame()
    abnormal = anomaly_score > _EIF_THRESHOLD_
    df["abnormal"] = abnormal
    df["score"] = anomaly_score
    h2o.shutdown()
    return df


def plot_anomaly(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(df['counts'], df['cost'], df['suspicion'], c='grey', s=40)
    ax.scatter(df['counts_norm'], df['cost_norm'], df['risk_norm'], c='grey', s=40)
    ax.view_init(600, 600)
    plt.show()



def query_raw_data():
    return []


def query_data_mapping():
    return []

def abnormal_detection_3D(start_hour=1, end_hour=0):
    print('start anomaly detection:', start_hour, end_hour)
    data = query_raw_data()
    df = pd.DataFrame(data)
    df = df.replace({'NaN': 0})
    df = df.replace({'Infinity': 0})
    print('read data rows:', len(df.index))

    # df = add_real_fraud_data(df)

    # df['counts_norm'] = (df['counts'] - df['counts'].min()) / (df['counts'].max() - df['counts'].min())
    # df['cost_norm'] = (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min())
    # df['suspicion_norm'] = (df['suspicion'] - df['suspicion'].min()) / (df['suspicion'].max() - df['suspicion'].min())

    df['counts_norm'] = (df['call_counts'] - df['call_counts'].mean()) / (df['call_counts'].std())
    df['cost_norm'] = (df['cost_sum'] - df['cost_sum'].mean()) / (df['cost_sum'].std())
    df['suspicion_norm'] = (df['suspicion_avg'] - df['suspicion_avg'].mean()) / (df['suspicion_avg'].std())
    df['risk_norm'] = (df['risk_avg'] - df['risk_avg'].mean()) / (df['risk_avg'].std())

    df = iforest_detection_h2o(df)
    plot_anomaly(df)
    abnormal_df = df[df['abnormal']]
    #abnormal_df = abnormal_df[(abnormal_df['counts'] >= 15) | (abnormal_df['cost'] >= 50)]
    print(tabulate(abnormal_df, headers='keys', tablefmt='psql', showindex=False))
    abnormal_df['call_in_out'] = None
    now = int(time.time() * 1000)
    abnormal_df = abnormal_df.loc[now - abnormal_df.t < 20*60*1000]
    abnormal_df = abnormal_df.loc[abnormal_df.call_counts >= _MIN_CALLS_]
    abnormal_df = abnormal_df.loc[abnormal_df.cost_sum > _MIN_COST_]
    for index, row in abnormal_df.iterrows():
        if row['id'] not in abnormal_detect_cache:
            call_in_out = query_data_mapping()
            in_out_dict = {}
            for i in call_in_out:
                in_out_dict[
                    i['report_identifiers_wxcCallingCountry'] + '(' + i['report_identifiers_wxcCallingCountryCode'] + ')->' + i['report_identifiers_wxcCalledCountry'] + '(' + i['report_identifiers_wxcCalledCountryCode'] + ')']  = \
                    i['count(*)']
            abnormal_df.at[index, 'call_in_out'] = in_out_dict
            abnormal_detect_cache[row['id']] = 1
        else:
            continue
    abnormal_df = abnormal_df[abnormal_df['call_in_out'].notna()]
    country_col = abnormal_df.pop('call_in_out')
    abnormal_df.insert(1, 'call_country', country_col)
    abnormal_df.rename(columns={'id': 'userId'}, inplace=True)
    abnormal_df.rename(columns={'t': 'callStartTime'}, inplace=True)
    message = 'Anomaly detected:\n```\n' + tabulate(abnormal_df, headers='keys', tablefmt='psql',
                                 showindex=False) + '\n```'
    if len(abnormal_df) > 0:
        send_msg()
        abnormal_df_list = abnormal_df.to_dict(orient='records')
        send_kafka()


def send_msg():
    pass


def send_kafka():
    pass

def verify_history_data():
    df = pd.read_csv("data/training/final_training_dataset.csv")
    h2o.init()
    predictors = ["cost_sum", "call_counts", "suspicion_sum", "risk_sum"]
    #predictors = ["risk_sum", "risk_min", "suspicion_avg", "risk_avg", "cost_max", "call_counts"]
    eif = H2OExtendedIsolationForestEstimator(model_id="eif.hex",
                                              ntrees=100,
                                              sample_size=256,
                                              extension_level=len(predictors) - 1)
    eif.train(x=predictors,
              training_frame=h2o.H2OFrame(df))
    eif_result = eif.predict(h2o.H2OFrame(df))
    anomaly_score = eif_result["anomaly_score"].as_data_frame()
    abnormal = anomaly_score > _EIF_THRESHOLD_
    df["abnormal"] = abnormal
    df["score"] = anomaly_score
    h2o.shutdown()
    orig_fraud = df[df['fraud'] > 0]
    elf_detected = df[(df['score'] > 0.6) & (df['fraud'] > 0) & (df['call_counts'] > 15) & (df['batch'].notnull())]
    elf_false = df[(df['score'] > 0.6) & (df['fraud'] < 1) & (df['call_counts'] > 15) & (df['batch'].notnull())]
    print('all fraud:', len(orig_fraud.index))
    print('elf fraud:', len(elf_detected.index), 'elf false:', len(elf_false.index))
    print('')


if __name__ == '__main__':
    verify_history_data()
