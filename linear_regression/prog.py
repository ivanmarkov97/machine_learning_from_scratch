from matplotlib import pyplot as plt
import numpy as np


def generate_data_1d(num_points, x_from, x_to, angle, noise):
	X = np.linspace(x_from, x_to, num_points)
	y = X * angle + np.random.rand(num_points) * noise
	return X, y


class LinearRegression:

	def fit(self, X, y):
		"""
		Y = WX
		W = (X^T*X)^(-1)*X^T*Y
		"""
		self.X = X
		self.y = y
		self.y0 = 0

		print('stat learn')

		if len(self.X.shape) > 1: #N-dims
			self.X = self.X.T
			self.W = np.dot(
				np.dot(
					np.linalg.pinv(
						np.dot(self.X.T,self.X)
					), self.X.T),
				self.y
			)
		else: #1-dim
			self.W = np.dot(
				np.dot(1 / (np.dot(self.X.T, self.X)), self.X.T), self.y
			)


	def fit_with_grad_1d(self, X, y, epoches, learn_rate):
		"""
		out = bx
		loss = 1/2M * sum(y - bx)^2
		d(loss)/db = 1/M * sum((y - (y0 + bx))*-x)
		d(loss)/dy0 = 1/M * sum((y - (y0 + bx))*-1)
		b = b - d(loss)/db * rate
		y0 = y0 - 
		"""
		self.X = X
		self.y = y
		self.n = len(X)
		self.W = np.random.rand()
		self.y0 = np.random.rand()
		self.learn_rate = learn_rate

		print('grad learn')

		for _ in range(epoches):
			y_pred = self.W * self.X + self.y0
			dloss_dw = 1 / self.n * sum((y - y_pred) * (-self.X))
			dloss_dy0 = 1 / self.n * sum((y - y_pred) * (-1.0))

			self.W = self.W - dloss_dw * learn_rate
			self.y0 = self.y0 - dloss_dy0 * learn_rate


	def predict(self, X):
		if len(self.W.shape) > 1:
			return np.dot(self.W, X)
		return X * self.W + self.y0



if __name__ == '__main__':
	X, y = generate_data_1d(100, 0, 10, 3.0, 3.0)
	
	clf = LinearRegression()
	clf.fit(X, y)

	clf2 = LinearRegression()
	clf2.fit_with_grad_1d(X, y, epoches=10, learn_rate=0.01)

	X_test = np.linspace(0, 10, 10)
	y_test = clf.predict(X_test)

	X_test2 = np.linspace(0, 10, 10)
	y_test2 = clf2.predict(X_test2)

	plt.scatter(X, y)
	plt.plot(X_test, y_test, c='red')
	plt.plot(X_test2, y_test2, c='green')
	plt.show()
	
	print(clf.W)
	print(clf2.W)
