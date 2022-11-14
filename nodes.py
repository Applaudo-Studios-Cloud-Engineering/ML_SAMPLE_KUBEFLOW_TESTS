from kfp.v2.dsl import component, Dataset, OutputPath, InputPath, Input, Output
from pandas import DataFrame

import mlflow
import mlflow.sklearn

# from typing import List, Dict, Any

@component(
    packages_to_install=["pandas", "fsspec", "gcsfs", "numpy"],
)
def create_dataset(path: str, dataset: OutputPath(Dataset)):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path)

    # drop_unnecessary_columns
    columns = ['PassengerId']

    df = df.drop(columns, axis=1)

    # fill_empty_age_values
    mean = df['Age'].mean()
    std = df['Age'].std()
    total_nulls = df['Age'].isnull().sum()

    randon_age_range = np.random.randint(mean - std, mean + std, size=total_nulls)
    age_feat_slice = df['Age'].copy()
    age_feat_slice[np.isnan(age_feat_slice)] = randon_age_range

    df['Age'] = age_feat_slice
    df['Age'] = df['Age'].astype(int)

    # fill_empty_embarked_values
    common_val = 'S'

    df['Embarked'] = df['Embarked'].fillna(common_val)

    df['Fare'] = df['Fare'].fillna(0)
    df['Fare'] = df['Fare'].astype(int)

    # with open(test, 'w') as writer:
    #    writer.write(df.to_csv())
    df.to_csv(dataset, index=False)


@component(
    packages_to_install=["pandas", "fsspec", "gcsfs", "numpy"],
)
def create_feature_engineering_pipeline(path: InputPath(Dataset),
                                        dataset_feature_engineering: OutputPath(Dataset)) -> Dataset:
    import pandas as pd
    import re

    df = pd.read_csv(path)

    # create_deck_feature
    drop_cabin = False

    decks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df['Cabin'] = df['Cabin'].fillna('U0')
    df['Deck'] = df['Cabin'].apply(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(decks)
    df['Deck'] = df['Deck'].fillna(0)
    df['Deck'] = df['Deck'].astype(int)

    if drop_cabin:
        df.drop(['Cabin'], axis=1)

    # create_title_feature
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    drop_name = False

    if drop_name:
        df = df.drop(['Name'], axis=1, inplace=True)

    # encode_sex
    sexes = {"male": 0, "female": 1}
    df['Sex'] = df['Sex'].map(sexes)

    # create_relatives_feature
    df['Relatives'] = df['SibSp'] + df['Parch']

    drop_features = False
    if drop_features:
        df = df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # drop_unnecessary_columns
    df = df.drop(['Cabin', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)

    # encode_embarked_ports
    encoded_ports = {'S': 0, 'C': 1, 'Q': 2}

    df['Embarked'] = df['Embarked'].map(encoded_ports)

    # encode_fare
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
    df.loc[df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

    # encode_age_ranges
    df['Age'] = df['Age'].astype(int)

    df.loc[df['Age'] <= 11, 'Age'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
    df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4
    df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5
    df.loc[(df['Age'] > 40) & (df['Age'] <= 66), 'Age'] = 6
    df.loc[df['Age'] > 66, 'Age'] = 7

    # encode_title_feature
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    df['Title'] = df['Title'].map(titles)
    df['Title'] = df['Title'].fillna(0)

    # create_age_class_feature
    df['Age_Class'] = df['Age'] * df['Pclass']

    df.to_csv(dataset_feature_engineering, index=False)


@component(
    packages_to_install=["pandas", "scikit-learn", "mlflow"],
)
def create_ml_pipeline_classifier(path: InputPath(Dataset)) -> float:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import mlflow

    mlflow.set_experiment(experiment_name="mlflow-demo")

    df = pd.read_csv(path)

    # split_dataset_for_training
    x = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # create_and_train_decision_tree_model
    model = DecisionTreeClassifier()
    model.fit(x, y)

    # compute_accuracy
    accuracy = model.score(x, y)
    print(accuracy)
    accuracy = round(accuracy * 100, 2)

    mlflow.log_metric("accuracy", accuracy)

    return accuracy


@component(
    packages_to_install=["pandas", "scikit-learn"],
)
def create_ml_pipeline_regressor(path: InputPath(Dataset)) -> float:
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    df = pd.read_csv(path)

    # split_dataset_for_training
    x = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # create_and_train_decision_tree_model
    model = DecisionTreeRegressor()
    model.fit(x, y)

    # compute_accuracy
    accuracy = model.score(x, y)
    print(accuracy)
    accuracy = round(accuracy * 100, 2)

    return accuracy


@component(
    packages_to_install=["pandas", "scikit-learn"],
)
def create_ml_pipeline_extra_classifier(path: InputPath(Dataset)) -> float:
    import pandas as pd
    from sklearn.tree import ExtraTreeClassifier

    df = pd.read_csv(path)

    # split_dataset_for_training
    x = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # create_and_train_decision_tree_model
    model = ExtraTreeClassifier()
    model.fit(x, y)

    # compute_accuracy
    accuracy = model.score(x, y)
    print(accuracy)
    accuracy = round(accuracy * 100, 2)

    return accuracy


@component(
    packages_to_install=["pandas", "scikit-learn"],
)
def create_ml_pipeline_extra_regressor(path: InputPath(Dataset)) -> float:
    import pandas as pd
    from sklearn.tree import ExtraTreeRegressor

    df = pd.read_csv(path)

    # split_dataset_for_training
    x = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # create_and_train_decision_tree_model
    model = ExtraTreeRegressor()
    model.fit(x, y)

    # compute_accuracy
    accuracy = model.score(x, y)
    print(accuracy)
    accuracy = round(accuracy * 100, 2)

    return accuracy


@component()
def evaluate_accuracy(first_acc: float,second_acc: float,third_acc: float,fourth_acc: float) -> str:
    if first_acc >= second_acc and first_acc >= third_acc and first_acc >= fourth_acc:
        return 'DecisionTreeClassifier ' + str(first_acc)
    elif second_acc >= first_acc and second_acc >= third_acc and second_acc >= fourth_acc:
        return 'DecisionTreeRegressor ' + str(second_acc)
    elif third_acc >= first_acc and third_acc >= second_acc and third_acc >= fourth_acc:
        return 'DecisionTreeRegressor ' + str(third_acc)
    return 'ExtraTreeRegressor ' + str(fourth_acc)
