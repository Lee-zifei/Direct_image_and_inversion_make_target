import torch
import numpy as np


class AverageMeter:
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))
	return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def snr(output, target):
	eps = 1e-5
	results = 0
	batch_size = target.size(0)
	for i in range(batch_size):
		loss1 = sum(sum((target[i, 0, :, :] - output[i, 0, :, :]) ** 2)) + eps
		loss2 = sum(sum(target[i, 0, :, :] ** 2)) + eps
		results += 10 * torch.log(loss2 / loss1)
	return results / batch_size