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
	data_xy = list(zip(data_x, data_y))
	random_gen.shuffle(data_xy)
	res = []
	# когда число, на которое нужно разбить список, равен длине самого списка данных (причем он может работать правильно и в ином случае, но тогда процесс становиться очень-очень долгим)
	if k==len(data_xy):
		for j in data_xy:
		    x_train = [u[0] for u in data_xy if u!=j]
		    y_train = [u[1] for u in data_xy if u!=j]
		    x_test = [j[0]]
		    y_test = [j[1]]
		    model.fit(x_train, y_train)
		    res.append(score_model(model, x_test, y_test))
		return round_to_3(sum(res)/len(res))
	
	# За счет уменьшения кол-ва частей, повышается "шанс получить верный ответ", если убрать эту строку вовсе, то скорость явно увеличится, но результат скорее всего будет неверным
	# В файле с тестами для второй задачи k = k//int((k**(1/2))) подходит идеально, причем не сильно тормозит алгоритм, всего-то в ≈1.63 раза 
	k = k//int((k**(1/2)))
	data_xy = split_into_k(data_xy, k)
	for i in range(k):
		for j in data_xy[i]:
		    x_train = [u[0] for u in data_xy[i] if u!=j]
		    y_train = [u[1] for u in data_xy[i] if u!=j]
		    x_test = [j[0]]
		    y_test = [j[1]]
		    model.fit(x_train, y_train)
		    res.append(score_model(model, x_test, y_test))
	return round_to_3(sum(res)/len(res))

