import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pprint import pprint

import src.models
from src.pot import pot_eval, calc_point2point
from src.plotting import plot_labels

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def save_model(folder, model, optimizer, scheduler, epoch, accuracy_list, name=''):
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model{name}.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dataset, dims, window_size, d_model=None, 
			   test=False, checkpoints_path=None, loss='MSE', forecasting=False):
	""" Load or create a model with the specified parameters.
		Parameters:
		modelname (str): The name of the model class to be loaded or created.
		dims (int): The dimensions of the input data, corresponds to nb of features.
		window_size (int): The window size for the model.
		d_model (int, optional): The dimension of the model. Default is None.
		test (bool, optional): Whether to test the model. Default is False.
		checkpoints_path (str, optional): The path to load the model from. Default is None.
		loss (str, optional): The loss function to be used. Default is 'MSE'.
		forecasting (bool, optional): Whether to use forecasting mode. Default is False.

		Returns:
		tuple: A tuple containing the model, optimizer, scheduler, epoch, and accuracy_list.
		If a pre-trained model exists at the specified path and retraining is not required, 
		the model, optimizer, and scheduler states are loaded from the checkpoint. 
		Otherwise, a new model is created and initialized.
	"""

	model_class = getattr(src.models, modelname)
	if modelname == 'iTransformer':
		model = model_class(dims, window_size, d_model, forecasting).double()
	elif modelname == 'Transformer':
		model = model_class(dims, window_size, d_model).double()
	elif modelname == 'AnomalyTransformer':
		model = model_class(dims, window_size, d_model).double()
	else:
		model = model_class(dims, window_size).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	if checkpoints_path is not None:
		fname = os.path.join(checkpoints_path, 'model_best.ckpt')
	else:
		fname = f'{modelname}_{dataset}/window_size{window_size}/checkpoints/model_best.ckpt'
	if os.path.exists(fname) and test:
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC} from {fname}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

class EarlyStopper:
	""" Early stopping class to stop training when the validation loss does not improve.
		Parameters:
		patience (int, optional): The number of epochs to wait before stopping. Default is 2.
		min_delta (float, optional): The minimum change in validation loss to be considered an improvement. Default is 0.
	"""
	def __init__(self, patience=2, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_validation_loss = None
	
	def early_stop(self, validation_loss):
		if self.min_validation_loss is None:
			self.min_validation_loss = validation_loss
		elif validation_loss <= (self.min_validation_loss + self.min_delta):
			if validation_loss > self.min_validation_loss:
				self.min_validation_loss = validation_loss
			self.counter += 1
			if self.counter >= self.patience:
				return True
		else:
			self.min_validation_loss = validation_loss
			self.counter = 0
		return False
	

def loss_quantile(output, target, quantile):   
	"""pinball loss function, metric to asses accuracy of quantile predictions"""
	z = target - output
	loss = torch.where(z < 0, quantile * z, (quantile - 1) * z)
	return loss

class combined_loss(nn.Module):
	""" Combined loss function for the iTransformer model.
		Parameters:
		window_size (int): The window size for the model.
		penalty (bool, optional): Whether to include a variability penalty term in the loss function. Default is False.

		The combined loss function consists of three components: Huber loss, quantile loss at 0.25, and quantile loss at 0.75.
		The penalty term is included if the penalty parameter is set to True.
	"""
	def __init__(self, window_size, penalty=False):
		super(combined_loss, self).__init__()
		self.window_size = window_size
		self.penalty = penalty
	
	def forward(self, output, target):
		output1, output2, output3 = torch.split(output, self.window_size, dim=1)
		loss_Huber = torch.nn.HuberLoss(reduction='none')
		huber = loss_Huber(output1, target)
		quantile1 = loss_quantile(output2, target, quantile=0.25)
		quantile2 = loss_quantile(output3, target, quantile=0.75)
		penalty = torch.log( torch.abs(output3 - output2) + 1e-4 )
		if self.penalty:
			loss_tot = huber + quantile1 + quantile2 - 0.01*penalty
		else:
			loss_tot = huber + quantile1 + quantile2
		# print('huber:', huber.mean().item(), 'quantile1:', quantile1.mean().item(), 'quantile2:', quantile2.mean().item(), 'penalty:', penalty.mean().item())
		# print('loss:', loss_tot.mean().item())
		return loss_tot

def local_pot(loss, lossT, labels, q=1e-5, plot_path=None):
	# computes POT for each variate separately
	df_res_local = pd.DataFrame()
	preds = []
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]  	
		result_local, pred = pot_eval(lt, l, ls, plot_path, f'dim{i}', q=q)
		preds.append(pred)
		df_res = pd.DataFrame.from_dict(result_local, orient='index').T
		df_res_local = pd.concat([df_res_local, df_res], ignore_index=True)
	preds = np.array(preds).T
	preds = preds.astype(int)
	return preds, df_res_local

def local_anomaly_labels(preds, labels, q=1e-5, plot_path=None, nb_adim=1):
	"""
	Calculate local anomaly labels based on the predicted scores and true labels.
	Parameters:
		preds (numpy.ndarray): Predicted scores for each dimension.
		labels (numpy.ndarray): True labels for each dimension.
		q (float): Threshold for anomaly detection.
		plot_path (str): Path to save the plots.
		nb_adim (int): Number of anomalous dimensions to consider.
	Returns:
		labelspred (numpy.ndarray): Predicted labels based on the anomaly scores.
		result_local1 (dict): Dictionary containing various evaluation metrics.
	"""

	labelspred = (np.sum(preds, axis=1) >= nb_adim) + 0

	if plot_path is not None:
		plot_labels(plot_path, f'labels_adim{nb_adim}', y_pred=labelspred, y_true=labels)
	
	result_local = calc_point2point(predict=labelspred, actual=labels)
	result_local1 = {'f1': result_local[0], 'precision': result_local[1], 'recall': result_local[2], 
				'TP': result_local[3], 'TN': result_local[4], 'FP': result_local[5], 'FN': result_local[6], 
				'ROC/AUC': result_local[7], 'MCC': result_local[8]}
	result_local1.update({'detection level q': q})
	print(f'{color.HEADER}Local results with {nb_adim} anomalous dimensions for anomaly{color.ENDC}')
	pprint(result_local1)
	return labelspred, result_local1

# loss and lr adjustment used for AnomalyTransformer
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
