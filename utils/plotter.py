import util
import seaborn as sns
import pandas as pd
import numpy as np

def plot_approx_posterior(cov, means, index):
	mean = means[index]
	mean, cov = util.product_gaussians(mean, np.zeros(2), cov, np.identity(2))
	data = np.random.multivariate_normal(mean, cov, 200)
	df = pd.DataFrame(data, columns=["x", "y"])
	xlim = (mean[0] - 3*np.sqrt(cov[0][0]),mean[0] + 3*np.sqrt(cov[0][0]))
	ylim = (mean[1] - 3*np.sqrt(cov[1][1]),mean[1] + 3*np.sqrt(cov[1][1]))
	sns.jointplot(x="x", y="y", data=df, kind="kde", stat_func= None, xlim = xlim, ylim = ylim)	

def plot_exact_posterior(observed, sigx, W, index):
	x = observed[index].transpose()

	a = sigx*np.identity(3)
	temp = np.identity(2) + np.dot(np.dot(W.transpose(), np.linalg.inv(a)), W)
	temp = np.linalg.inv(temp)
	temp = np.dot(temp, W.transpose())
	temp = np.dot(temp, np.linalg.inv(a))
	mean = np.dot(temp,x)
	temp = np.dot(W.transpose(), np.linalg.inv(a))
	cov = np.linalg.inv(np.identity(2) + np.dot(temp, W))
		
	data = np.random.multivariate_normal(mean, cov, 20000)

	df = pd.DataFrame(data, columns=["x", "y"])
	xlim = (mean[0] - 3*np.sqrt(cov[0][0]),mean[0] + 3*np.sqrt(cov[0][0]))
	ylim = (mean[1] - 3*np.sqrt(cov[1][1]),mean[1] + 3*np.sqrt(cov[1][1]))
	sns.jointplot(x="x", y="y", data=df, kind="kde", stat_func = None, xlim = xlim, ylim = ylim)