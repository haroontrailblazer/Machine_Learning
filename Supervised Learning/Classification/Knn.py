from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.4)

k_range=range(1,20)
accuracies =[]
for k in k_range:
    model=KNN(n_neighbors=k)
    model.fit(X_train,y_train)
    pre =model.predict(X_test)
    Acu=accuracy_score(y_test,pre)
    accuracies.append(Acu)

plt.style.use("dark_background")
plt.title("KNN: Varying Number of Neighbors")
plt.plot(k_range,accuracies,marker='o',markerfacecolor='blue')
plt.grid(True, alpha=0.1)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.show()

MN=int(input("Enter the number of neighbors you want to use: "))
model=KNN(n_neighbors=MN)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test,predictions))

# Visualize the test set predictions using PCA for dimensionality reduction
pca=PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)
plt.scatter(X_test_2d[:,0],X_test_2d[:,1],c=predictions,edgecolor='k',s=100,cmap='coolwarm')
plt.title(f"KNN Predictions with K={MN}")
plt.show()