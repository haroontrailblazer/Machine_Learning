from sklearn.linear_model import LinearRegression


reg = LinearRegression()


x = [[1], [2], [3], [4], [5], [6]]
y = [2, 2.5, 4.5, 3, 5, 4.7]


reg.fit(x, y)

S=reg.predict([[int(input("enter the number"))]])
print(S)
