from modules import *
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load iris data
iris_data = pd.read_csv('/Users/linlab2024/Desktop/temp_coding/aiPrinciple/irisdata/iris_header.csv')
X = iris_data.iloc[:, :-1].values  # Features
Y = iris_data.iloc[:, -1].values   # Target

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Train KNN model
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, Y_train)

# Predict and calculate accuracy
Y_predict = knn_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print('Accuracy Score:', score)

# Visualization
plt.figure(figsize=(6, 6))
colmap = np.array(['blue', 'green', 'red'])

# Scatter plot for actual test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=[colmap[int(i)] for i in Y_test], s=150, marker='o', alpha=0.5, label='Actual')

# Scatter plot for predicted test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=[colmap[int(i)] for i in Y_predict], s=50, marker='o', alpha=0.5, label='Predicted')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend()
plt.show()


# random generated data then split train & test data
"""
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

# plt.scatter(train_x, train_y)
# plt.scatter(test_x,test_y)
plt.scatter(x, y)
plt.show()
"""

# cluster hierarchy
"""
# Generate data
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

X = list(zip(x, y))
print(X)

# Perform hierarchical clustering
linkage_data = linkage(X, method='ward', metric='euclidean')

# Plot scatter plot
fig = plt.figure(figsize=(6, 6))
plt.scatter(x, y)
plt.show()

# Plot dendrogram
dendrogram(linkage_data)
plt.show()
"""
