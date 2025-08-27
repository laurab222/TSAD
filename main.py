import os, math
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
from pprint import pprint
from torchinfo import summary
from tslearn.metrics import SoftDTWLossPyTorch

from src.parse_args import parse_arguments
from src.pot import pot_eval
from src.plotting import plot_accuracies, plot_labels, plot_losses, plot_metrics, plotter, plotter2, compare_labels, plot_MSE_vs_ascore
from src.utils import load_model, save_model, local_anomaly_labels, local_pot, color, EarlyStopper, my_kl_loss, adjust_learning_rate
from src.diagnosis import hit_att, ndcg
from src.data_loader import MyDataset
from src.soft_dtw_cuda import SoftDTW


def backprop(epoch, model, data, feats, optimizer, scheduler, training=True, 
			 lossname='MSE', enc_feats=0, pred=False, forecasting=False):
	
	if 'OmniAnomaly' in model.name:
		l = nn.MSELoss(reduction = 'mean' if training else 'none')
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				d = d.view(-1)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			loss = []; y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
				d = d.view(-1)
				l1 = l(y_pred, d)
				loss.append(l1)
			y_pred = torch.stack(y_preds)
			loss = torch.stack(loss)
			y_pred = y_pred.view(-1, feats)
			loss = loss.view(-1, feats)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				d = d.view(-1)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			l1s = []
			datashape = feats * model.window_size
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
				d = d.view(-1)
				l1 = 0.1 * l(ae1, d) + 0.9 * l(ae2ae1, d)
				l1s.append(l1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			l1s = torch.stack(l1s)
			y_pred = ae1s[:, -feats:]
			loss = l1s[:, -feats:]
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d in data:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			loss = torch.empty(0)
			z_all = torch.empty(0)
			for d in data:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
				l1 = l(z, elem)[0]
				loss = torch.cat((loss, l1.view(-1, feats)), dim=0)
				if pred: z_all = torch.cat((z_all, z.view(-1, feats)), dim=0)
			if pred:
				return loss.detach().numpy(), z_all.detach().numpy()
			else:
				return loss.detach().numpy()
	elif model.name in ['Transformer', 'iTransformer']:
		if lossname == 'MSE':
			l = nn.MSELoss(reduction = 'none')
		elif lossname == 'Huber':
			l = nn.HuberLoss(reduction = 'none')
		elif lossname == 'softdtw':
			l = SoftDTW(use_cuda=torch.cuda.is_available(), normalize=True)
		elif lossname == 'softdtw_norm':
			l = SoftDTWLossPyTorch(normalize=True)
		else:
			raise ValueError('Loss function not implemented')
		n = epoch + 1
		if training:
			l1s = []
			for d in data: # d.shape is [B, window_size, N] or d = (dx, dy)
				if model.name == 'iTransformer' and model.forecasting:	
					elem = d[1][:, -1, :].view(-1, 1, feats)  # [B, 1, N]
					d = d[0]								  # [B, window_size, N]
				else:
					elem = d
				if enc_feats > 0:
					d_enc = d[:, :, :enc_feats]
					d = d[:, :, enc_feats:]
				else:
					d_enc = None
				# don't invert d because we have permutation later in DataEmbedding_inverted as part of model
				if model.output_attention:
					z = model(d, d_enc)[0]
				else:
					z = model(d, d_enc)
				l1 = l(z, elem)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			l = nn.MSELoss(reduction = 'none')  # always use MSE for testing
			z_all = torch.empty(0)
			loss = torch.empty(0)
			for i, d in enumerate(data): # d.shape is [B, window_size, N]
				if model.name == 'iTransformer' and model.forecasting:	
					elem = d[1][:, -1, :].view(-1, 1, feats)  # [B, 1, N]
					d = d[0]								  # [B, window_size, N]
				else:
					elem = d
				if enc_feats > 0:
					d_enc = d[:, :, :enc_feats]
					d = d[:, :, enc_feats:]
				else:
					d_enc = None
				if model.output_attention:
					z = model(d, d_enc)[0]
				else:
					z = model(d, d_enc)
				l1 = l(z, elem)
				loss = torch.cat((loss, l1), dim=0)
				if pred:
					z_all = torch.cat((z_all, z), dim=0)
			loss = loss.view(-1, feats)
			if pred:
				z_all = z_all.reshape(-1, feats)
				return loss.detach().numpy(), z_all.detach().numpy()  
			else:
				return loss.detach().numpy()
	elif 'AnomalyTransformer' in model.name:
		l = nn.MSELoss(reduction='mean')
		param_k = 3 # hyperparameter for AnomalyTransformer
		n = epoch + 1
		loss1_list = []
		iter_count = 0
		if training:
			for d in data:
				iter_count += 1
				output, series, prior, _ = model(d)
                # calculate Association discrepancy
				series_loss = 0.0
				prior_loss = 0.0
				for u in range(len(prior)):
					series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)).detach())) + torch.mean(
                        		my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)).detach(), series[u])))
					prior_loss += (torch.mean(my_kl_loss(
                        	(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)), series[u].detach())) 
							+ torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)))))
				series_loss = series_loss / len(prior)
				prior_loss = prior_loss / len(prior)
				rec_loss = l(output, d)
				loss1_list.append((rec_loss - param_k * series_loss).item())
				loss1 = rec_loss - param_k * series_loss
				loss2 = rec_loss + param_k * prior_loss
            # Minimax strategy
			optimizer.zero_grad()
			loss1.backward(retain_graph=True)
			loss2.backward()
			optimizer.step()
			# learning rate adjustment for AnomalyTransformer
			adjust_learning_rate(optimizer, epoch + 1, model.lr)	
			return np.average(loss1_list), optimizer.param_groups[0]['lr']
		else:
			l = nn.MSELoss(reduction='none')
			z_all = torch.empty(0)
			loss = torch.empty(0)
			temperature = 50
			attens_energy = []
			for d in data:
				local_bs = d.shape[0]
				output, series, prior, _ = model(d)
				if pred:
					z_all = torch.cat((z_all, output), dim=0)
				loss = torch.mean(l(d, output), dim=-1) # averages over features
				# loss = l(d, output)
				# loss = loss.view(-1, feats)
				series_loss = 0.0
				prior_loss = 0.0
				for u in range(len(prior)):
					if u == 0:
						series_loss = my_kl_loss(series[u], (
								prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)).detach()) * temperature
						prior_loss = my_kl_loss(
							(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)),
							series[u].detach()) * temperature
					else:
						series_loss += my_kl_loss(series[u], (
								prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)).detach()) * temperature
						prior_loss += my_kl_loss(
							(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, model.window_size)),
							series[u].detach()) * temperature
			
				metric = torch.softmax((-series_loss - prior_loss), dim=-1)
				cri = metric * loss
				cri = cri.detach().numpy()
				attens_energy.append(cri)
			
			attens_energy = np.concatenate(attens_energy, axis=None)
			test_energy = np.repeat(attens_energy[:, np.newaxis], repeats=feats, axis=1)
			if pred:
				z_all = z_all.reshape(-1, feats)
				return test_energy, z_all.detach().numpy()	
			else:
				return test_energy
	elif 'LSTM_AE' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1
		if training:
			l1s = []
			for d in data:
				y_pred = model(d)
				l1 = l(y_pred, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			loss = torch.empty(0)
			z_all = torch.empty(0)
			for d in data:
				y_pred = model(d)
				l1 = l(y_pred, d)
				loss = torch.cat((loss, l1), dim=0)
				z_all = torch.cat((z_all, y_pred), dim=0)
			loss = loss.view((-1, feats))
			z_all = z_all.view((-1, feats))
			if pred:
				return loss.detach().numpy(), z_all.detach().numpy()
			else:
				return loss.detach().numpy()
			

if __name__ == '__main__':
	args = parse_arguments()
	print('Arguments:\n')
	pprint(vars(args))
	print('\nCUDA available:', torch.cuda.is_available())
	print("MPS (Apple Silicon GPU) is available:", torch.backends.mps.is_available(), '\n')

	# define path for results, checkpoints & plots & create directories
	if args.name:
		folder = f'{args.model}/{args.model}_{args.dataset}/window{args.window_size}_steps{args.step_size}_dmodel{args.d_model}_feats{args.feats}_eps{args.epochs}_{args.loss}/{args.name}'
	else:
		folder = f'{args.model}/{args.model}_{args.dataset}/window{args.window_size}_steps{args.step_size}_dmodel{args.d_model}_feats{args.feats}_eps{args.epochs}_{args.loss}'
	plot_path = f'{folder}/plots'
	res_path = f'{folder}/results'
	if args.checkpoint is None:
		checkpoints_path = f'{folder}/checkpoints'
	else:
		checkpoints_path = args.checkpoint
	os.makedirs(plot_path, exist_ok=True)
	os.makedirs(res_path, exist_ok=True)

	# create data sets
	train = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='train', feats=args.feats, less=args.less, enc=args.enc, k=args.k, forecasting=args.forecasting)
	test = MyDataset(args.dataset, args.window_size, args.window_size, args.model, flag='test', feats=args.feats, less=args.less, enc=args.enc, k=-1, forecasting=args.forecasting)
	train_test = MyDataset(args.dataset, args.window_size, args.window_size, args.model, flag='train', feats=args.feats, less=args.less, enc=args.enc, k=-1, forecasting=args.forecasting)
	labels = test.get_labels()
	
	print('train shape with windows:', train.data.shape)
	print('test shape with windows:', test.data.shape)
	print('labels shape:', labels.shape)
	
	if args.k > 0:
		valid = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='valid', feats=args.feats, less=args.less, enc=args.enc, k=args.k, forecasting=args.forecasting)
		print(f'{args.k}-fold valid set', valid.__len__(), valid.data.shape)

	feats = train.feats
	enc_feats = train.enc_feats
	
	# load model
	model, optimizer, scheduler, epoch, accuracy_list = load_model(modelname=args.model,
																dataset=args.dataset, 
																dims=feats, 
																window_size=args.window_size, 
																d_model=args.d_model, 
																test=args.test, 
																checkpoints_path=checkpoints_path, 
																loss=args.loss,
																forecasting=args.forecasting)

	# Calculate and print the number of parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'total params: {total_params}, trainable params: {trainable_params}')

	# Create data loader
	data_loader_train = DataLoader(train, batch_size=model.batch, shuffle=False)
	data_loader_test = DataLoader(test, batch_size=model.batch, shuffle=False)
	data_loader_train_test = DataLoader(train_test, batch_size=model.batch, shuffle=False)
	if args.k > 0:
		data_loader_valid = DataLoader(valid, batch_size=model.batch, shuffle=False)

	# save arguments and additional info in config file
	with open(f'{folder}/config.txt', 'w') as f:
		f.write(f'{args.model} on {args.dataset}\n \n')
		f.write(str(args)+'\n')
		f.write(f'total params: {total_params}, trainable params: {trainable_params}\n')
		f.write(f'feats: {feats}\n')
		f.write(f'train: {train.__len__()}\n')
		if args.k > 0:
			f.write(f'valid: {valid.__len__()}\n')
		f.write(f'test: {test.__len__()}\n')
		f.write(f'ts_lengths train: {train.get_ts_lengths()}\n')
		if args.k > 0:
			f.write(f'ts_lengths valid: {valid.get_ts_lengths()}\n')
		f.write(f'ts_lengths test: {test.get_ts_lengths()}\n')
		f.write(f'optimizer: {optimizer}, \nscheduler: {scheduler}\n')

		sample_batch = next(iter(data_loader_train))
		f.write('\nModel Summary:\n')
		if model.name == 'TranAD':
			window = sample_batch.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, -1, feats)
			f.write(str(summary(model, input_data=[window, elem], depth=5, verbose=0)))
		else:
			f.write(str(summary(model, input_data=sample_batch, depth=5, verbose=0)))
		
	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		if args.k > 0:
			early_stopper = EarlyStopper(patience=5, min_delta=0.0)
			min_lossV = 100
		num_epochs = args.epochs; e = epoch + 1; start_time = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			model.train()
			lossT, lr = backprop(e, model, data_loader_train, feats, optimizer, scheduler, training=True, 
								lossname=args.loss, enc_feats=enc_feats, forecasting=args.forecasting)
			if args.k > 0:
				model.eval()
				lossV = backprop(e, model, data_loader_valid, feats, optimizer, scheduler, training=False, 
					 			lossname=args.loss, enc_feats=enc_feats, forecasting=args.forecasting)
				lossV = np.mean(lossV)
				if lossV < min_lossV:
					min_lossV = lossV
					save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_best')
			else:
				lossV = 0
			tqdm.write(f'Epoch {e},\tL_train = {lossT}, \t\tL_valid = {lossV}, \tLR = {lr}')
			accuracy_list.append((lossT, lossV, lr))
			# save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, f'_epoch{e}')
			if args.k > 0 and early_stopper.early_stop(-lossV):
				print(f'{color.HEADER}Early stopping at epoch {e}{color.ENDC}')
				break
		train_time = time() - start_time
		print(f'{color.BOLD}Training time: {"{:10.4f}".format(train_time)} s or {"{:.2f}".format(train_time/60)} min {color.ENDC}')
		save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_final')
		if not os.path.exists(f'{checkpoints_path}/model_best.ckpt'):
			save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_best')
		plot_accuracies(accuracy_list, plot_path)
		plot_losses(accuracy_list, plot_path)
		np.save(f'{checkpoints_path}/accuracy_list.npy', accuracy_list)

	### Testing phase
	torch.zero_grad = True
	if args.k > 0:  # if using validation set, make sure to load best model
		checkpoint = torch.load(f'{checkpoints_path}/model_best.ckpt')
		model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

	### Scores
	lossT = backprop(-1, model, data_loader_train_test, feats, optimizer, scheduler, training=False, 
				  	lossname=args.loss, enc_feats=enc_feats, pred=False, forecasting=args.forecasting)  # need anomaly scores on training data for POT
	loss, y_pred = backprop(-1, model, data_loader_test, feats, optimizer, scheduler, training=False, 
						 lossname=args.loss, enc_feats=enc_feats, pred=True, forecasting=args.forecasting)

	if feats <= 30:
		testOO = test.get_complete_data_wpadding()
		nolabels = np.zeros_like(loss)
		plotter(plot_path, testOO, y_pred, loss, nolabels, test.get_ideal_lengths(), name='output_padded')
	
	# print(f'check data shapes before removing padding:'
	#    f'\ntrain loss {lossT.shape}, test loss{loss.shape}, labels {labels.shape}\n')

	if ('iTransformer' in model.name or model.name in ['LSTM_AE', 'Transformer', 'AnomalyTransformer']) and not args.forecasting:
		# cut out the padding from test data, loss tensors
		lossT_tmp, loss_tmp, y_pred_tmp = [], [], []
		print(test.get_ts_lengths(), np.sum(test.get_ts_lengths()), len(test.get_ts_lengths()))
		print(test.get_ideal_lengths(), np.sum(test.get_ideal_lengths()), len(test.get_ideal_lengths()))
		start = 0
		for i, l in enumerate(test.get_ts_lengths()):
			loss_tmp.append(loss[start:start+l])
			y_pred_tmp.append(y_pred[start:start+l])
			start += test.get_ideal_lengths()[i]
		
		start = 0
		for i, l in enumerate(train_test.get_ts_lengths()):
			lossT_tmp.append(lossT[start:start+l])
			start += train_test.get_ideal_lengths()[i]

		lossT = np.concatenate(lossT_tmp, axis=0)
		loss = np.concatenate(loss_tmp, axis=0)
		y_pred = np.concatenate(y_pred_tmp, axis=0)

	# print(f'Check data shapes after removing padding:'
	#    f'\ntrain loss {lossT.shape}, test loss{loss.shape}, labels {labels.shape}\n')
	train_loss = np.mean(lossT)
	test_loss = np.mean(loss)

	# compress true labels to 1D
	true_labels = (np.sum(labels, axis=1) >= 1) + 0

	# plot time series with padding
	if feats <= 30:
		testO = test.get_complete_data()
		plotter(plot_path, testO, y_pred, loss, labels, test.get_ts_lengths(), name='output')

	plot_MSE_vs_ascore(plot_path, loss, true_labels)

	### anomaly labels
	preds, df_res_local = local_pot(loss, lossT, labels, args.q, plot_path)
	true_labels = (np.sum(labels, axis=1) >= 1) + 0
	# local anomaly labels
	labelspred, result_local1 = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=1)  # inclusive OR
	majority = math.ceil(labels.shape[1] / 2)	# do majority voting over dimensions for local results instead of inclusive OR
	labelspred_maj, result_local2 = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=majority)
	labelspred_all = []
	results_all = pd.DataFrame()

	# define anomaly starting from i anomalous dimensions, scan over i (between 1 and feats)
	if feats > 20:
		nb_adim = np.unique(np.concatenate((np.arange(1, 11), np.arange(20, feats, 10), [feats])))
	else:
		nb_adim = np.arange(1, feats+1)
	for i in nb_adim:
		lpred, res = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=i)
		labelspred_all.append(lpred)
		results_all = pd.concat([results_all, pd.DataFrame.from_dict(res, orient='index').T], ignore_index=True)
	

	# global anomaly labels
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	true_labels = (np.sum(labels, axis=1) >= 1) + 0
	result_global, pred2 = pot_eval(lossTfinal, lossFinal, true_labels, plot_path, f'all_dim', q=args.q)
	labelspred_glob = (pred2 >= 1) + 0
	result_global.update(hit_att(loss, labels))
	result_global.update(ndcg(loss, labels))
	if not args.test:
		result_global.update({'train_time': train_time})
	result_global.update({'detection_level_q': args.q})
	result_global.update({'train_loss': train_loss, 'test_loss': test_loss})
	print(f'{color.HEADER}Global results with {feats} anomalous dimensions for anomaly{color.ENDC}') 
	pprint(result_global)

	# more plots
	plot_labels(plot_path, 'labels_global', y_pred=labelspred_glob, y_true=true_labels)
	plot_metrics(plot_path, name=['local_incl_OR', 'local_maj_voting', 'global'], 
			y_pred=[labelspred, labelspred_maj, labelspred_glob], y_true=true_labels)
	compare_labels(plot_path, pred_labels=[labelspred, labelspred_maj, labelspred_glob], true_labels=true_labels, 
				plot_labels=['Local anomaly\n(inclusive OR)', 'Local anomaly\n(majority voting)', 'Global'], 
				name='_all')
	
	if feats <= 40:
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred, labels, name='_local_or')
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred_maj, labels, name='_local_maj')
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred_glob, labels, name='_global')

	### saving results
	# results for different anomaly definitions, scan over i dimensions (between 1 and feats)
	results_all.index = nb_adim  
	results_all.to_csv(f'{res_path}/res_local_all.csv')

	# results when using individual dimensions for anomaly labels, incl. OR, majority voting or average to combine them
	df_res_global = pd.DataFrame.from_dict(result_global, orient='index').T
	df_res_global.index = ['global']
	result_local1 = pd.DataFrame.from_dict(result_local1, orient='index').T
	result_local2 = pd.DataFrame.from_dict(result_local2, orient='index').T
	result_local1.index = ['local_all']
	result_local2.index = ['local_all_maj']
	df_res = pd.concat([df_res_local, result_local1, result_local2, df_res_global])
	df_res.to_csv(f'{res_path}/res.csv')	

	# labels predicted for 3 different anomaly label methods
	df_labels = pd.DataFrame( {'local_all': labelspred, 'local_all_maj': labelspred_maj, 'global': labelspred_glob} )
	df_labels.to_csv(f'{res_path}/pred_labels.csv', index=False)
