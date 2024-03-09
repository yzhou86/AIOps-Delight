import json
import operator

import pandas as pd
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


xgb_clf = xgb.XGBClassifier()
xgb_booster = xgb.Booster()
xgb_booster.load_model('model/wxc_xgb.model')
xgb_clf._Booster = xgb_booster


def get_dataset():
    df = pd.read_csv("data/training/final_training_dataset.csv")
    df_f = df[df['fraud'] > 0]
    df_l = df[df['fraud'] < 1]
    print('Total fraud samples: ', len(df_f.index), df_f['call_counts'].sum())
    print('Total legal samples: ', len(df_l.index), df_l['call_counts'].sum())
    dfx = shuffle(df)
    dfx.reset_index(drop=True, inplace=True)
    X = dfx.iloc[:, 1:-1]
    y = dfx['fraud']
    return X, y


def calc_accuracy(xgb_model, X_test, y_test):
    y_predict = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)
    predictions = [round(value) for value in y_predict]
    X_test['fraud'] = y_test
    X_test['predict'] = predictions
    ignore_false = X_test[(X_test['call_counts'] < 15) & (X_test['fraud'] < 1) & (X_test['predict'] > 0)]
    print('ignored false alert (call_count<15):', len(ignore_false.index))
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    recall = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    return accuracy, recall, confusion

def xgb_cv():
    X, y = get_dataset()
    xgbc_model = xgb.XGBClassifier()
    print("XGBoost raw：", cross_val_score(xgbc_model, X, y, cv=5).mean())

    split_ratio = 0.2
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=10)
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print("（mean accuracy = (TP+TN)/(P+N) ）")
    print("xgboost：", xgb_model.score(x_test, y_test))
    print('evaluation')
    print(metrics.classification_report(y_test, y_pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

    param_test1 = {
        'n_estimators': [i for i in range(50, 300, 50)],
        'max_depth': [i for i in range(3, 10, 2)],
        'min_child_weight': [i for i in range(1, 6, 2)],
        'colsample_bytree': [i/10 for i in range(1, 10, 1)],
        'colsample_bylevel': [i/10 for i in range(1, 10, 1)],
        'learning_rate': [0.1, 0.01, 0.05, 0.001]
    }
    from sklearn.model_selection import GridSearchCV
    custom_scorer = metrics.make_scorer(metrics.recall_score, pos_label=1)
    gsearch = GridSearchCV(
        estimator=xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=140,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=300,
            seed=27),
        param_grid=param_test1,
        scoring=custom_scorer,
        n_jobs=4,
        cv=5)
    gsearch.fit(x_train, y_train)
    print('max_depth_min_child_weight')
    print('gsearch1.best_params_', gsearch.best_params_)
    print('gsearch1.best_score_', gsearch.best_score_)


def xgb_params():
    X, y = get_dataset()

    max_recall = 0
    max_acc = 0
    best_param = None
    # for estimator in range(50, 500, 50):
    # for depth in range(3, 10, 1):
    # for weight in range(1, 10, 1):
    # for colsample_bylevel in range(1, 10, 1):
    #     colsample_bylevel = colsample_bylevel / 10
    for i in range(1, 2, 1):
        xgb_model = xgb.XGBClassifier(n_estimators=450,
                                      gamma=0,
                                      max_depth=6,
                                      min_child_weight=2,
                                      colsample_bytree=0.2,
                                      colsample_bylevel=0.1,
                                      subsample=0.9,
                                      reg_lambda=0,
                                      reg_alpha=0,
                                      seed=33,
                                      objective="binary:logistic",
                                      learning_rate=0.0001,
                                      random_state=42,
                                      eval_metric="auc",
                                      nthread=-1,
                                      scale_pos_weight=300)
        recalls = []
        accs = []
        for r_state in range(1, 100, 10):
            split_ratio = 0.2
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=r_state)

            xgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

            acc_test, recall, confusion_test = calc_accuracy(xgb_model, x_test, y_test)
            print('test dataset accuracy: ', acc_test)
            print('test dataset recall: ', recall)
            print(confusion_test)
            recalls.append(recall)
            accs.append(acc_test)
        avg_recall = sum(recalls) / len(recalls)
        avg_acc = sum(accs) / len(accs)
        print('avg_recall:', avg_recall, 'avg_acc', avg_acc)
    #     if avg_recall > max_recall:
    #         max_recall = avg_recall
    #         best_param = estimator
    # print('best:', best_param, max_recall)
    #     if avg_acc > max_acc:
    #         max_acc = avg_acc
    #         best_param = colsample_bylevel
    print('best:', best_param, max_acc)


def get_model_by_recall(recall=True):
    if not recall:
        xgb_model = xgb.XGBClassifier(n_estimators=450,
                                      gamma=0,
                                      max_depth=10,
                                      min_child_weight=2,
                                      colsample_bytree=0.2,
                                      colsample_bylevel=0.1,
                                      subsample=0.9,
                                      reg_lambda=0,
                                      reg_alpha=0,
                                      seed=33,
                                      objective="binary:logistic",
                                      learning_rate=0.1,
                                      random_state=42,
                                      eval_metric="auc",
                                      nthread=-1,
                                      scale_pos_weight=300)

    else:
        # complex model to cover only fraud calls in dataset, but less false alert in future
        xgb_model = xgb.XGBClassifier(n_estimators=450,
                                      gamma=0,
                                      max_depth=6,
                                      min_child_weight=2,
                                      colsample_bytree=0.2,
                                      colsample_bylevel=0.1,
                                      subsample=0.9,
                                      reg_lambda=0,
                                      reg_alpha=0,
                                      seed=33,
                                      objective="binary:logistic",
                                      learning_rate=0.0001,
                                      random_state=42,
                                      eval_metric="auc",
                                      nthread=-1,
                                      scale_pos_weight=300)
    return xgb_model


def release():
    X, y = get_dataset()
    xgb_model = get_model_by_recall(True)
    xgb_model.fit(X, y, eval_set=[(X, y)], verbose=False)

    xgb_model.save_model('model/wxc_xgb.model')
    acc_test, recall, confusion_test = calc_accuracy(xgb_model, X, y)
    print(acc_test, recall)
    print(confusion_test)

    feature_important = xgb_model.get_booster().get_score(importance_type='weight')
    feature_importance = dict(sorted(feature_important.items(), key=operator.itemgetter(1), reverse=True))
    print(json.dumps(feature_importance, indent=4))
    return xgb_model


def predict(X):
    y_predict = xgb_clf.predict(X)
    return y_predict


def predict_proba(X):
    y_predict = xgb_clf.predict_proba(X)
    return y_predict


def verify_data():
    data_df = pd.read_csv('data/cpm/20240130/calling-alerts-202312-202401.csv')
    culprits = set(data_df['culprit'])

    df = pd.read_csv("data/training/20240130/all_alert_dataset.csv", low_memory=False)
    data_df = df[df['culprit'].isin(culprits)]

    X = data_df.iloc[:, 1:-2]
    proba_result = predict_proba(X)
    predict_result = predict(X)
    data_df['predict_fraud'] = predict_result.tolist()
    data_df['proba'] = proba_result.tolist()
    print(data_df)
    data_df.to_csv('data/training/20240130/verify_result.csv', index=False)

    data_list = data_df.to_dict('records')
    xgb_alert = 0
    xgb_miss = 0
    xgb_confirm = 0
    for data_item in data_list:
        if data_item['predict_fraud'] > 0:
            xgb_alert = xgb_alert + 1
        if data_item['predict_fraud'] < 1 and data_item['fraud'] > 0:
            xgb_miss = xgb_miss + 1
        if data_item['predict_fraud'] > 0 and data_item['fraud'] > 0:
            xgb_confirm = xgb_confirm + 1
    print('xgb alert:', xgb_alert, ',confirm:', xgb_confirm, ',miss:', xgb_miss)


if __name__ == '__main__':
    #xgb_params()
    # release()
    # verify_data()
    xgb_cv()
    print('')

