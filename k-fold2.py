import pandas as pd
import random
from sklearn.linear_model import LinearRegression
class ShuffleOnceRandom():
    def __init__(self, seed=None):
        import random
        
        self._random_gen = random.Random(seed)
        self._shuffle_cnt = 0
        
    def shuffle(self, l):
        if self._shuffle_cnt > 0:
            raise RuntimeError('Нельзя использовать функцию shuffle больше одного раза')
            
        self._shuffle_cnt += 1
        self._random_gen.shuffle(l)
def split_into_k(l, k):
    l_mod_k = len(l) % k
    l_div_k = len(l) // k
    res = []
    for i in range(k):
        res.append(l[i * l_div_k:(i + 1) * l_div_k])
    for i in range(l_mod_k):
        res[i].append(l[l_div_k * k + i]) 
    return res
def score_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    res = 0
    for i in range(len(y_test)):
        res += abs(y_pred[i] - y_test[i])
    return res / len(y_test)
def round_to_3(x):    
    return round(x, 3)


def k_fold_solution(model, data_x, data_y, k, random_gen):
	# По окончании каждой итерации будем добавлять значение score_model в этот список. В самом конце найдем среднее арифметическое
	res = []

	# Из рекомендации
	data_xy = list(zip(data_x, data_y))
	random_gen.shuffle(data_xy)

	# Возвращаем все перемешанные данные обратно по своим переменным
	data_x = [i[0] for i in data_xy]
	data_y = [i[1] for i in data_xy]

	# Разделяем данные на k частей
	data_x, data_y = split_into_k(data_x, k), split_into_k(data_y, k)

	# Проходимся по данным, тут x и y это x_test и y_test, они входить в обучающую сборку не будут
	for x,y in zip(data_x, data_y):

		# Находим x_train проходясь по data_x и выбирая все элементы из списка, 
		# которые не равны текущему элементу в цикле (в данном случае x), и также убираем у него лишние квадратные скобки
		x_train = [i[0] for i in data_x if i!=x]

		# Находим y_train проходясь по data_y и выбирая все элементы из списка,
		# которые не равны текущему элементу в цикле (в данном случае y), и также убираем у него лишние квадратные скобки
		y_train = [i[0] for i in data_y if i!=y]

		# Задаем тестовую сборку новыми переменными, просто для удобства, а по факту переменные можно и не создавать
		x_test = x
		y_test = y
		
		# обучаем модель
		model.fit(x_train, y_train)

		# заносим оцененную точность модели в списка res
		res.append(score_model(model, x_test, y_test))

	# создаем переменную куда записываем результат, а именно среднее арифметическое всего списка
	result = round_to_3(sum(res)/len(res))
	# выводим среднее арифметическое всего списка
	print(result)
	# возвращаем среднее арифметическое всего списка
	return result



# Добавил тестирование сразу, чтобы код можно было запустить
def k_fold_test():
    data_x_example_1 = [[1], [2], [3]]
    data_y_example_1 = [1, 2, 3]
    
    assert k_fold_solution(LinearRegression(), 
                           data_x_example_1, data_y_example_1, 3, 
                           ShuffleOnceRandom(0)) == 0.0
    
    data_example_2 = pd.read_csv('k_fold_test_data.csv')
    
    data_x_example_2 = [[x] for x in list(data_example_2['x'])]
    data_y_example_2 = list(data_example_2['y'])
    
    assert k_fold_solution(LinearRegression(), 
                           data_x_example_2, data_y_example_2, 100, 
                           ShuffleOnceRandom(0)) == 0.602
    
    print('Тест прошёл успешно!')

k_fold_test()
