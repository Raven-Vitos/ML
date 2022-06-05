import numpy as np


# Функция сигмойда
# Необходима для определения значения весов
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
    
    
# Входные данные
x = np.array([[0, 0, 1, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])
              

# Ожидаемый результат             
y = np.array([[0.0, 1.0, 1.0, 0.0]]).T


np.random.seed(1)


# Инициализация веса случайным образом со средним 0
syn0 = 2 * np.random.random((4, 1)) - 1

l1 = []


for iter in range(10000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))
    
    # На сколько мы ошиблись?
    l1_error = y - l1
    
    
    # Перемножим это с наклоном сигмойды
    # на основе значений в l1
    l1_delta = l1_error * sigmoid(l1, True)
    
    # Обновим веса
    syn0 += np.dot(l0.T, l1_delta)
    
print("Выходные данные после обучения:")
print(l1)

new_one = np.array([0, 1, 0, 1])
l1_new = sigmoid(np.dot(new_one, syn0))
print("Новые данные:")
print(l1_new)