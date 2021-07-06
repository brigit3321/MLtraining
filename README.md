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
![image](https://user-images.githubusercontent.com/78330497/124572771-e2145f00-de83-11eb-9a67-642a49675fb4.png)

# 入力値と目標値の検証
```python
print(f'train score: {model.score(x_train, t_train)}')
print(f'train score: {model.score(x_test, t_test)}')
```
![image](https://user-images.githubusercontent.com/78330497/124573185-47685000-de84-11eb-8cfd-68ff9c2bd861.png)

# 推論
```python
y = model.predict(x_test)
print(f'予測値: {y[0]}')
print(f'目標値: {t_test[0]}')

print(f'予測値: {y[1]}')
print(f'目標値: {t_test[1]}')
```
![image](https://user-images.githubusercontent.com/78330497/124573761-ccec0000-de84-11eb-848d-3bf5032ab79f.png)

![image](https://user-images.githubusercontent.com/78330497/124573844-dd9c7600-de84-11eb-8764-d45784797ebb.png)

・重回帰分析とは２つ以上のデータから回帰分析を行うことである
・今回使用したデータは入力値と目標値の検証の結果あまり良い数値は出ていない

## 続いて決定木の実装を行う

```python
dataset = load_iris()
columns_name = dataset.feature_names
x = dataset.data
t = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
model.fit(x_train, t_train)
```

```python
print(f'train score:{model.score(x_train, t_train)}')
print(f'test score:{model.score(x_test, t_test)}')
```
ここまでで入力変数と目標値を入力
![image](https://user-images.githubusercontent.com/78330497/124575578-7d0e3880-de86-11eb-94ad-86d8478cc0dd.png)

決定木の図を記載する
```python
model.predict(x_test)
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(model)
graph_tree = graphviz.Source(dot_data)
graph_tree
```
![image](https://user-images.githubusercontent.com/78330497/124576222-0faed780-de87-11eb-9903-63f0a15692c9.png)

入力変数の影響度を測る
```python
feature_importance = model.feature_importances_
feature_importance
y = columns_name
width = feature_importance
plt.barh(y=y, width=width)
```
![image](https://user-images.githubusercontent.com/78330497/124576510-57356380-de87-11eb-824c-7c10baf5f58e.png)

・決定木のメリット
必要な前処理が少ないの扱いやすい
・決定木のデメリット
過学習になりやすいのでハイパーパラメータが大事になってくる

# SVCの実装
```python
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, t_train)
print(f'train score:{model.score(x_train, t_train)}')
print(f'test score:{model.score(x_test, t_test)}')
```
![image](https://user-images.githubusercontent.com/78330497/124577478-433e3180-de88-11eb-8a79-9ee4952fae1e.png)

# 標準化
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
```
# 標準化したデータを元に再度データを実装する
```python
model_std = SVC()
model_std.fit(x_train_std, t_train)
print(f'train score:{model.score(x_train, t_train)}')
print(f'test score:{model.score(x_test, t_test)}')
print('==================')
print(f'train score:{model_std.score(x_train_std, t_train)}')
print(f'test score:{model_std.score(x_test_std, t_test)}')
```
![image](https://user-images.githubusercontent.com/78330497/124578285-07f03280-de89-11eb-89bb-839b27f3c299.png)

・SVMは２つのカテゴリを識別する分類器

# ハイパーパラメータの調整法（グリッドサーチ）
```python
from sklearn.model_selection import GridSearchCV
estimator = DecisionTreeClassifier(random_state=0)
param_grid = [{
    'max_depth': [3, 20, 50],
    'min_samples_split': [3, 20, 30]
}]
cv = 5
tuned_model = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=cv,
                          return_train_score=False)
tuned_model.fit(x_train_val, t_train_val)
pd.DataFrame(tuned_model.cv_results_).T
```
![image](https://user-images.githubusercontent.com/78330497/124579424-0d9a4800-de8a-11eb-837d-ccdc524ab708.png)
```python
tuned_model.best_params_
best_model = tuned_model.best_estimator_
print(best_model.score(x_train_val, t_train_val))
print(best_model.score(x_test, t_test))
```
ハイパーパラメータはMLにおいて非常に重要な値になる、
今回はその中でグリッドサーチという方法で実装してみた。

・メリット
ある程度もれなくハイパーパラメータを網羅できる
・デメリット
場合によってたくさんの組み合わせをするため、
計算に時間がかかる場合もある。









