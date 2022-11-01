import argparse
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import r2_score, mean_squared_error


parser = argparse.ArgumentParser('Tree demo')
parser.add_argument('--method', type=str, choices=['ranFor', 'ExtTree', 'DecTree', 'SVR'], default='SVR')
parser.add_argument('--train_prob', type=int, default=20)
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--scale', type=int, default=10000)
args = parser.parse_args()

def Machine_learning_IRD(file_name, method):
	read = np.loadtxt(file_name, delimiter=',', skiprows=1)[:, 1:(1+args.input_size)]
	x_data = read[0:read.shape[0]-1, :]
	y_data = read[1:read.shape[0], :]
	tr_x = x_data[0:x_data.shape[0]-args.train_prob, :]
	tr_y = y_data[0:x_data.shape[0]-args.train_prob, :]
	te_x = x_data[x_data.shape[0]-args.train_prob:, :]
	te_y = y_data[x_data.shape[0]-args.train_prob:, :]
	print('tr_x shape:', tr_x.shape, 'tr_y shape:', tr_y.shape, 'te_x shape:', te_x.shape, 'te_y shape:', te_y.shape)

	if method == 'ranFor':
		parameter_space = {
			"n_estimators": [10, 50, 100],
			"min_samples_leaf": [2, 4, 6],
			"min_samples_split": [3, 6, 9],
		}
		clf = GridSearchCV(RandomForestRegressor(),param_grid=parameter_space)
	elif method == 'ExtTree':
		parameter_space = {
			"n_estimators": [10, 50, 100],
			"max_features": [1, 2, 3],
		}
		clf = GridSearchCV(ExtraTreesRegressor(), param_grid=parameter_space)
	elif method == 'DecTree':
		parameter_space = {
			"max_depth": [2, 5, 10],
			"min_impurity_decrease": [0.002, 0.005, 0.01],
		}
		clf = GridSearchCV(DecisionTreeRegressor(), param_grid=parameter_space)

	clf.fit(tr_x, tr_y)
	y_pred = clf.predict(x_data)
	te_y_pred = clf.predict(te_x)
	print('Te-I-R2:', r2_score(te_y_pred[:, 0] * args.scale, te_y[:, 0] * args.scale))
	print('Te-R-R2:', r2_score(te_y_pred[:, 1] * args.scale, te_y[:, 1] * args.scale))
	print('Te-D-R2:', r2_score(te_y_pred[:, 2] * args.scale, te_y[:, 2] * args.scale))
	print('Te-I-MSE:', mean_squared_error(te_y_pred[:, 0] * args.scale, te_y[:, 0] * args.scale))
	print('Te-R-MSE:', mean_squared_error(te_y_pred[:, 1] * args.scale, te_y[:, 1] * args.scale))
	print('Te-D-MSE:', mean_squared_error(te_y_pred[:, 2] * args.scale, te_y[:, 2] * args.scale))

	#print('y_pred:', y_pred.shape, 'x_data', x_data.shape)
	fig = plt.figure(figsize=(12, 4), facecolor='white')
	ax_I = fig.add_subplot(131, frameon=False)
	ax_R = fig.add_subplot(132, frameon=False)
	ax_D = fig.add_subplot(133, frameon=False)
	plt.show(block=False)

	ax_I.cla()
	ax_I.set_title('I')
	ax_I.set_xlabel('t')
	ax_I.set_ylabel('I(t)')
	ax_I.plot(range(y_pred.shape[0]), y_pred[:, 0]*args.scale, 'b--')
	#ax_I.scatter(range(y_pred.shape[0]), y_pred[:, 0] * args.scale, s=5, c='b')
	ax_I.plot(range(y_data.shape[0]), y_data[:, 0]*args.scale, 'g-')
	#ax_I.scatter(range(y_pred.shape[0]), x_data[:, 0] * args.scale, s=5, c='g')

	ax_R.cla()
	ax_R.set_title('R')
	ax_R.set_xlabel('t')
	ax_R.set_ylabel('R(t)')
	ax_R.plot(range(y_pred.shape[0]), y_pred[:, 1]*args.scale, 'b--')
	ax_R.plot(range(y_data.shape[0]), y_data[:, 1]*args.scale, 'g-')

	ax_D.cla()
	ax_D.set_title('D')
	ax_D.set_xlabel('t')
	ax_D.set_ylabel('D(t)')
	ax_D.plot(range(y_pred.shape[0]), y_pred[:, 2]*args.scale, 'b--')
	ax_D.plot(range(y_data.shape[0]), y_data[:, 2]*args.scale, 'g-')

	fig.tight_layout()
	plt.savefig(file_name[:-4] + method +'.png')
	np.savetxt(file_name[:-4] + method + '.csv', y_pred, delimiter=',')

	print('I-R2:', r2_score(y_pred[:, 0]*args.scale, y_data[:, 0]*args.scale))
	print('R-R2:', r2_score(y_pred[:, 1]*args.scale, y_data[:, 1]*args.scale))
	print('D-R2:', r2_score(y_pred[:, 2]*args.scale, y_data[:, 2]*args.scale))
	print('I-MSE:', mean_squared_error(y_pred[:, 0]*args.scale, y_data[:, 0]*args.scale))
	print('R-MSE:', mean_squared_error(y_pred[:, 1]*args.scale, y_data[:, 1]*args.scale))
	print('D-MSE:', mean_squared_error(y_pred[:, 2]*args.scale, y_data[:, 2]*args.scale))

file = ['Italy_IRD.csv', 'USA_IRD.csv', 'Columbia_IRD.csv', 'South_africa_IRD.csv', 'wuhan_IRD.csv', 'Piedmont_IRD.csv', 'AU_NSW_IRD.csv', 'AU_VIC_IRD.csv']
methods = ['ranFor', 'ExtTree', 'DecTree']
for i in range(len(file)):
	for j in range(len(methods)):
		print(file[i], methods[j])
		Machine_learning_IRD(file[i],methods[j])
