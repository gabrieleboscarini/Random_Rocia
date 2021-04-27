import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = pd.read_csv('C:/Users/gabro/Desktop/Tesi2021/datasets/db_perugia_immunofenotipi.csv')

train_features, test_features, train_labels, test_labels = train_test_split(
    data.drop(labels=['Target','Patient'], axis=1),
    data['Target'],
    test_size=0.25,
    random_state=42)

correlated_features = set()
correlation_matrix = data.corr()


for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

prediction_labels = rf.predict(test_features)

print(confusion_matrix(test_labels,prediction_labels))
print(classification_report(test_labels,prediction_labels))
print(accuracy_score(test_labels, prediction_labels))



















