import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("1000_Companies.csv")

label = LabelEncoder()
df["State"] = label.fit_transform(df["State"])

# sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.show()

X = df.drop("Profit", axis=1)
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r_squared = model.score(X_test, y_test)
print(f"R-squared value: {r_squared}")

#98.3 accuracy
