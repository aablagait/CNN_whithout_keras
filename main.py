import tensorflow as tf
import cv2 as cv
import numpy as np
import os


# создаю функцию сигмоиды, для настройки весов
def sigmoid(x, der=False):
	if der:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

# записываю названия файлов с изображениями в список
bee = os.listdir(path='bee')
wasp = os.listdir(path='wasp')

# объявляют пыстые списки, в которые
# буду записывать тензоры изображений
bee_tensor = []
wasp_tensor = []

# в цикле перебираю 80% изображений
# и записываю их тензоры в список
for i in range(int(len(bee) // 1.2)):
	bee_name = cv.imread('bee/' + str(bee[i]))
	bee_name = bee_name / 255
	bee_name = cv.resize(bee_name, (16, 16))
	bee_name = bee_name.flatten()
	bee_tensor.append(bee_name)

for i in range(int(len(wasp) // 1.2)):
	wasp_name = cv.imread('wasp/' + str(wasp[i]))
	wasp_name = wasp_name / 255
	wasp_name = cv.resize(wasp_name, (16, 16))
	wasp_name = wasp_name.flatten()
	wasp_tensor.append(wasp_name)

# создаю два списка, чтобы каждому элементы входного списка
# соответствовало нужное значение в выходном
x_list = []
y_list = []
for i in range(len(wasp_tensor)):
	x_list.append(bee_tensor[i])
	y_list.append(0)

for i in range(len(wasp_tensor)):
	x_list.append(wasp_tensor[i])
	y_list.append(1)


x = np.array(x_list)

y = np.array([y_list]).T
np.random.seed(1)
syn0 = 2 * np.random.random((768, 1)) - 1
l1 = []
# процесс обучения на 10000 итераций
for iter in range(10000):
	l0 = x
	l1 = sigmoid(np.dot(l0, syn0))
	l1_error = y - l1
	l1_delta = l1_error * sigmoid(l1, True)
	syn0 += np.dot(l0.T, l1_delta)

# вывод результатов обучения
print("Выходные данные после тренеровки:")
print(l1)
# проверка на незнакомых данных
new_one = np.array(bee_tensor[1800])
l1_new = sigmoid(np.dot(new_one, syn0))
print('новые данные')
print(l1_new)