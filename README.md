# 重回帰分析の実装

ボストン近郊の住宅データを取得
```python 
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
t = dataset.target
columns = dataset.feature_names
df = pd.DataFrame(x, columns=columns)
df.head()
df['Target'] = t
df.head()
t = df['Target'].values
x = df.drop(labels=['Target'], axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
```
ここまでで入力変数と目標値を入力

# ここから重回帰分析に入ります
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, t_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
model.coef_
```
# パラメータを可視化
```python
plt.figure(figsize=(10, 7))
plt.bar(x=columns, height=model.coef_)
```

