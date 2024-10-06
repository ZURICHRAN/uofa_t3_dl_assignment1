import numpy as np
from sklearn.model_selection import train_test_split


# 定义感知器模型类
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # 初始化权重为0
        self.bias = 0  # 初始化偏置为0

        # 感知器的训练过程
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # 更新权重和偏置
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def _activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_function(linear_output)


class PerceptronWithRegularization:
    def __init__(self, learning_rate=0.01, n_iters=1000, l1_ratio=0.0, l2_ratio=0.0):
        """
        l1_ratio: 控制 L1 正则化强度 (0 表示无 L1 正则化)
        l2_ratio: 控制 L2 正则化强度 (0 表示无 L2 正则化)
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 开始迭代训练
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # 计算更新步长
                update = self.lr * (y[idx] - y_predicted)

                # 更新权重 (加入正则化项)
                if self.l1_ratio > 0:
                    # L1 正则化，惩罚权重的绝对值
                    self.weights += update * x_i - self.lr * self.l1_ratio * np.sign(self.weights)
                if self.l2_ratio > 0:
                    # L2 正则化，惩罚权重的平方
                    self.weights += update * x_i - self.lr * self.l2_ratio * self.weights
                if self.l1_ratio == 0 and self.l2_ratio == 0:
                    # 无正则化
                    self.weights += update * x_i

                # 更新偏置
                self.bias += update

    def _activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_function(linear_output)


# 加载数据并解析
X = []
y = []

with open('diabetes_scale.txt', 'r') as file:
    for line in file:
        line_data = line.strip().split()  # 按空格分隔数据
        label = float(line_data[0])  # 获取标签 (第1列)
        features = np.zeros(8)  # 假设有8个特征
        for item in line_data[1:]:
            index, value = item.split(':')  # 将特征索引和值分开
            features[int(index) - 1] = float(value)  # 填充特征数组
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 实例化感知器并进行训练
# perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
# perceptron.fit(X_train, y_train)
#
# # 预测并计算测试集上的准确率
# predictions = perceptron.predict(X_test)
# accuracy = np.mean(predictions == y_test)
# print(f"模型的准确率: {accuracy:.2f}")

# 改变学习率并观察结果
for lr in [0.1, 0.01, 0.001, 0.0001]:
    for n_iters in [1, 10, 100, 1000]:
        perceptron = Perceptron(learning_rate=lr, n_iters=n_iters)
        perceptron.fit(X_train, y_train)
        predictions = perceptron.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"学习率: {lr},迭代次数: {n_iters}, 准确率: {accuracy:.2f}")

# 改变迭代次数并观察结果
for n_iters in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]:
    perceptron = Perceptron(learning_rate=0.01, n_iters=n_iters)
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"学习率: {0.01},迭代次数: {n_iters}, 准确率: {accuracy:.2f}")

# 仅使用 L1 正则化
for l1 in np.arange(0.001, 0.011, 0.001):
    perceptron_l1 = PerceptronWithRegularization(learning_rate=0.01, n_iters=1000, l1_ratio=l1)
    perceptron_l1.fit(X_train, y_train)
    predictions = perceptron_l1.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"使用 L1 正则化,L1 ratio:{l1},模型的准确率: {accuracy:.2f}")

# 仅使用 L2 正则化
for l2 in np.arange(0.001, 0.011, 0.001):
    perceptron_l2 = PerceptronWithRegularization(learning_rate=0.01, n_iters=1000, l2_ratio=l2)
    perceptron_l2.fit(X_train, y_train)
    predictions = perceptron_l2.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"使用 L2 正则化,L2 ratio:{l2},模型的准确率: {accuracy:.2f}")
