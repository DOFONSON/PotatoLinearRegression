import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('данные/Картофель_цена.csv', sep=';', decimal=',')
data['Месяц'] = pd.to_datetime(data['Месяц'], format='%b.%y')
data['Месяц_число'] = (data['Месяц'].dt.year - data['Месяц'].dt.year.min()) * 12 + data['Месяц'].dt.month


X = data[['Месяц_число', 'Инфляция в стране']]
y = data['Цена']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


model = LinearRegression()
model.fit(X_train, y_train)



month = int(input('Введите через сколько месяцев вы хотите узнать цену на картофель '))

infl = float(input('Введите средний процент инфляции за пройденное время '))
    


predicted_price = model.predict([[56 + month, infl]])  
print(f'Предсказанная цена на картофель: {predicted_price[0]:.2f}')


plt.scatter(X['Месяц_число'], y, color='blue', label='Исходные данные')
plt.xlabel('Месяц_число')
plt.ylabel('Цена')
plt.title('Линейная регрессия для цены картофеля')


plt.plot(X['Месяц_число'], model.predict(X), color='red', linewidth=2, label='Линейная регрессия')
plt.legend(loc='upper left')
plt.show()