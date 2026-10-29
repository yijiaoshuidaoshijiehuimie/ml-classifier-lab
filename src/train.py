import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 读取sklearn自带葡萄酒数据集
from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

X = df.drop("target", axis=1)
y = df["target"]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练随机森林分类器
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"模型准确率：{acc:.4f}")

# 保存模型到项目根目录
joblib.dump(model, "model.pkl")
print("模型保存完成 model.pkl")
