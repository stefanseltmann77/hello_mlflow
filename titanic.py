import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_dataframes():
    """creating simple train- and testing-data"""
    df_raw = pd.read_csv('train.csv')
    df_features = pd.concat([pd.get_dummies(df_raw.Sex, 'gender'),
                             pd.get_dummies(df_raw.Pclass, 'pclass'),
                             pd.get_dummies(df_raw.Embarked, 'Embarked'),
                             df_raw[['SibSp', 'Fare']]], axis=1)
    return train_test_split(df_features, df_raw.Survived, test_size=0.33, random_state=42)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataframes()
    params = {'n_estimators': 20}

    # setup mlflow and register the logging parts in a context manager
    mlflow.set_experiment('titanic')
    with mlflow.start_run():
        rfc = RandomForestClassifier(**params)
        rfc.fit(X_train, y_train)
        auc_score = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])

        mlflow.log_params(params)
        mlflow.log_metric("auc_score", auc_score)


    # check the results via GUI, if local server is started.
    experiment_id = mlflow.get_experiment_by_name('titanic').experiment_id
    df_runs = mlflow.search_runs(experiment_id)
    mlflow.get_tracking_uri()