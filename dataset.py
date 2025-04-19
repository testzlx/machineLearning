import numpy as np
import pandas as pd

np.random.seed(42)  # 保证结果可复现

# 生成特征数据：100 行 × 7 列，值在一定范围内
X = np.random.uniform(low=0, high=10, size=(100, 7))

# 从列中选取有影响的特征列：X2, X4, X5, X6
X2 = X[:, 2]
X4 = X[:, 4]
X5 = X[:, 5]
X6 = X[:, 6]

# 构造目标值 y，加上高斯噪声
noise = np.random.normal(0, 1, 100)
y = 5 * X2 + 3 * np.sin(X4) - 2 * X5 + 0.5 * X6**2 + noise

# 打包成 DataFrame
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(7)])
df['y'] = y
pd.set_option('display.max_rows', 100)

# 如果你想看所有列也可以设置一下
pd.set_option('display.max_columns', 10)
print(df.to_string(index=False))