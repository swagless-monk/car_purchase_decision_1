import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

cars_df = pd.read_csv('../data/datasets/car_purchase_decision.csv')

print(cars_df.head())

"""
Output col: Purchased - 0 = no; 1 = yes
Input col: Gender Int - 1 = male; 2 = female
"""

X = cars_df.drop(columns=['User ID', 'Purchased', 'Gender'])
y = cars_df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = DTC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

scores = accuracy_score(y_test, predictions)
print(scores)

joblib.dump(model, 'car_purchase_decision.joblib')