# syntax for linear regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()


x = [[1], [2], [3], [4], [5], [6]]
y = [2, 2.5, 4.5, 3, 5, 4.7]


model.fit(x, y)
S=model.predict([[int(input("enter the number"))]])
print(S)