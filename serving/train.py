import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train(path: str = "model.joblib") -> float:
    d = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, stratify=d.target, random_state=0)
    m = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xtr, ytr)
    joblib.dump({"model": m, "classes": list(d.target_names)}, path)
    return float(m.score(Xte, yte))


if __name__ == "__main__":
    print("test accuracy:", round(train(), 3))
