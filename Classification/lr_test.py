import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r'C:\Users\2026\Desktop\ML\DataSet\Social_Network_Ads.csv')
print('Dataset head:')
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('X_test:\n', X_test)
print('X_train:\n', X_train)
