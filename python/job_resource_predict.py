import json
import time

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from job_resource_collect import read_job_history_usage_flink, read_job_history_usage_spark_exec, send_kafka


def predict_forecast(Y_df, freq, season_length, horizon):
    models = [
        AutoARIMA(season_length=season_length)
    ]
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1
    )
    Y_hat_df = sf.forecast(df=Y_df, h=horizon)
    return Y_hat_df


def predict_forecast_job_resource(timestamp, measurement, freq, season_length, horizon, threshold, env):
    print('start resource forecast job for:', measurement)
    if measurement == 'flink':
        data = read_job_history_usage_flink(env)
    else:
        data = read_job_history_usage_spark_exec(env)
    messages = list()
    print('got job total:', len(data.keys()))

    print('start build df')
    raw_dfs = []
    for key, value in data.items():
        print('start to check job:', key)

        number_key = key
        df = value
        df = df[~df.applymap(lambda x: isinstance(x, str) and 'Infinity' in x).any(axis=1)]
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['ds'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cpu_df = df[['ds', 'cpu_load']].copy()
        cpu_df.rename(columns={'cpu_load': 'y'}, inplace=True)
        cpu_df = data_process(cpu_df)
        cpu_df['unique_id'] = number_key + '_cpu'
        mem_df = df[['ds', 'mem_load']].copy()
        mem_df.rename(columns={'mem_load': 'y'}, inplace=True)
        mem_df = data_process(mem_df)
        mem_df['unique_id'] = number_key + '_mem'
        raw_dfs.append(cpu_df)
        raw_dfs.append(mem_df)

    if len(raw_dfs) == 0:
        print('get job metrics null')
        return
    merged_df = pd.concat(raw_dfs)

    print('start to predict on merged df')
    predict_df = predict_forecast(merged_df, freq, season_length, horizon)
    predict_groups = predict_df.groupby('unique_id')

    for key, value in data.items():
        print('start to check job:', key)
        try:
            number_key = key
            df = value
            df = df[~df.applymap(lambda x: isinstance(x, str) and 'Infinity' in x).any(axis=1)]
            time_point_ratio = len(df)/((max(df['time'])-min(df['time']))/1000/60/10)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['ds'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            cpu_df = df[['ds', 'cpu_load']].copy()
            cpu_df.rename(columns={'cpu_load': 'y'}, inplace=True)
            cpu_df = data_process(cpu_df)
            cpu_df['unique_id'] = number_key
            cpu_forecast_df = predict_groups.get_group(number_key + '_cpu')
            mem_df = df[['ds', 'mem_load']].copy()
            mem_df.rename(columns={'mem_load': 'y'}, inplace=True)
            mem_df = data_process(mem_df)
            mem_df['unique_id'] = number_key
            mem_forecast_df = predict_groups.get_group(number_key + '_mem')
            task_count = df['task_count'].max()
            cpu_request = df['cpu_request'].max()
            mem_request = df['mem_request'].max()
            cpu_cost_save = 0
            mem_cost_save = 0
            new_request_cpu = cpu_request * cpu_forecast_df['AutoARIMA'].median() / threshold
            new_request_mem = mem_request * mem_forecast_df['AutoARIMA'].median() / threshold
            total_cpu_request = cpu_request * task_count
            total_mem_request = mem_request * task_count
            if new_request_cpu < 0.01:
                new_request_cpu = 0.01
            if new_request_cpu < cpu_request:
                cpu_cost_save = (cpu_request - new_request_cpu) * task_count
            if new_request_mem < mem_request:
                mem_cost_save = (mem_request - new_request_mem) * task_count
            cpu_forecast_df.rename(columns={'AutoARIMA': 'cpu_load'}, inplace=True)
            mem_forecast_df.rename(columns={'AutoARIMA': 'mem_load'}, inplace=True)
            result_df = cpu_forecast_df
            result_df.insert(2, 'mem_load', mem_forecast_df['mem_load'].values)
            tags = {
                "env": env,
                "cpu_cost_save": float(cpu_cost_save * time_point_ratio),
                "mem_cost_save": float(mem_cost_save * time_point_ratio),
                "new_cpu_request": float(new_request_cpu),
                "new_mem_request": float(new_request_mem),
                "cpu_request": float(cpu_request),
                "mem_request": float(mem_request),
                "total_cpu_request": float(total_cpu_request),
                "total_mem_request": float(total_mem_request),
                "task_count": int(task_count)
            }
            message = {
                "appId": "test_job_resource_monitor",
                "featureName": "test_unified_monitor",
                "metric_type": "test_job_cpu_mem_cost_save_metrics",
                "measurement": measurement,
                "number_key": number_key,
                "number_value": 0,
                "tags": tags,
                "timestamp": timestamp
            }
            messages.append(json.dumps(message).encode('utf-8'))
            for index, row in result_df.iterrows():
                t = int(row['ds'].timestamp() * 1000)
                cpu_v = row['cpu_load']
                mem_v = row['mem_load']
                tags = {
                    "env": env,
                    "cpu_load": cpu_v,
                    "mem_load": mem_v,
                }
                message = {
                    "appId": "test_job_resource_monitor",
                    "featureName": "test_unified_monitor",
                    "ints": t,
                    "metric_type": "test_job_cpu_mem_predict_metrics",
                    "measurement": measurement,
                    "number_key": number_key,
                    "number_value": 0,
                    "tags": tags,
                    "timestamp": timestamp
                }
                messages.append(json.dumps(message).encode('utf-8'))
        except Exception as e:
            print('predict one job fail: ', key, e)
    print('start to send kafka')
    send_kafka(messages)
    print('end of resource forecast job for:', measurement)


def data_process(df):
    df = df[df['y'] != 0]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df_resampled = df.resample('H').mean()
    # Fill na with the value in other day of same time hour
    for i in range(len(df_resampled)):
        if pd.isnull(df_resampled.iloc[i]['y']):
            hour = df_resampled.index[i].hour
            value = df_resampled[df_resampled.index.hour == hour]['y'].dropna()
            if len(value) > 0:
                df_resampled.iloc[i]['y'] = np.mean(value)
    # Fill na with the mean before and after
    df_resampled['y'] = df_resampled['y'].fillna((df_resampled['y'].bfill() + df_resampled['y'].ffill()) / 2)
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={'index': 'ds'})
    return df_resampled


def schedule_job(measurements, freq, season_length, horizon, threshold, env):
    timestamp = int(time.time() * 1000)
    print('start resource forecast schedule job: timestamp=', timestamp)
    for measurement in measurements:
        predict_forecast_job_resource(timestamp, measurement, freq, season_length, horizon, threshold, env)


if __name__ == '__main__':
    timestamp = int(time.time() * 1000)
    schedule_job(['spark'], '60T', 24, 24*7, 0.9, 'DFW')
    # schedule_job(['flink', 'spark'], '60T', 24, 24 * 7, 0.9, 'FRA')
    # schedule_job(['flink', 'spark'], '60T', 24, 24 * 7, 0.9, 'AWS')
