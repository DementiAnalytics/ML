import pandas as pd
from preprocess import create_labels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    DATA = "Data/"

    features_list, labels = create_labels(DATA)

    X = pd.DataFrame(features_list)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    coef = pd.Series(clf.coef_[0], index=X.columns)
    coef = coef.sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef.values, y=coef.index)
    plt.title("Feature Coefficients (Logistic Regression)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()