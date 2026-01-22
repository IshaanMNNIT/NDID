import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if __name__ == "__main__":

    with open("data/processed/decision_dataset.pkl", "rb") as f:
        X, y = pickle.load(f)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds)

    print("Validation F1:", f1)

    with open("data/processed/decider.pkl", "wb") as f:
        pickle.dump(clf, f)
