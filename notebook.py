from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, seed=42):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_adaboost(X_train, y_train, seed=42):
    model = AdaBoostClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_pipeline(model_type="rf", seed=42):
    X_train, X_test, y_train, y_test = load_data(seed=seed)

    if model_type == "rf":
        model = train_random_forest(X_train, y_train, seed=seed)
    elif model_type == "ab":
        model = train_adaboost(X_train, y_train, seed=seed)
    else:
        raise ValueError("model_type deve ser 'rf' ou 'ab'")

    return evaluate(model, X_test, y_test)
