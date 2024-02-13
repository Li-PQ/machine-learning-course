import matplotlib.pyplot as plt
import seaborn as sns #用于可视化图表
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a data set for analysis
x, y = make_regression(n_samples=500, n_features = 1, noise=25, random_state=0)
# 生成回归数据集，n_samples=500指定生成的数据点数量，n_features=1指定生成的特征数，表明这是一个一维回归问题
# noise=25指定数据中的噪声水平，random_state=0指定控制随机数生成的种子，确保代码的可重复性

# Split the data set into testing and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Create a linear regression object
regression = linear_model.LinearRegression() #生成一个线性回归的objective

# Train the model using the training set
regression.fit(x_train, y_train)

# Make predictions using the testing set
y_predictions = regression.predict(x_test)

# Plot the data
sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)
plt.plot(x_test, y_predictions, color='black')

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
