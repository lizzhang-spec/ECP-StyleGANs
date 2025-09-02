import argparse
import pickle
import math
import random
import os
import torch
from torch import nn
from torchvision import utils
from model import StyledGenerator
from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_network(ckpt):
	g_running = StyledGenerator(code_size).cuda()

	ckpt = torch.load(ckpt)
	g_running.load_state_dict(ckpt['g_running'])

	return g_running


def evaluate(Generator, fixed_noise, inception, real_acts, name):
	"""Custom GAN evaluation function"""

	step = int(math.log2(args.image_size)) - 2
	batch_size = 10

	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))

	images = []
	for i in range(gen_i):
		images.append(Generator(fixed_noise[i].cuda(), step=step, alpha=alpha).cpu())

	sample_path = f'sample/{name}/{str(0).zfill(6)}.png'
	utils.save_image(torch.cat(images, dim=0), sample_path, nrow=gen_i, normalize=True, value_range=(-1, 1))

	sample_num = args.sample_num
	fake_images, fake_acts = get_fake_images_and_acts(inception, Generator, code_size, step, alpha, sample_num,
													  batch_size)

	fid = compute_fid(real_acts, fake_acts)

	metrics = {'fid': fid}

	return metrics


if __name__ == '__main__':
	code_size = 512
	alpha = 1  

	parser = argparse.ArgumentParser(description='Test')

	parser.add_argument('--name', type=str, default='temp', help='name of experiment')

	parser.add_argument('--ckpt', type=str, default='./best.model', help='model')

	parser.add_argument('--image_size', default=512, type=int, help='image size')
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	parser.add_argument('--sample_num', default=500, type=int, help='number of samples for evaluation')

	args = parser.parse_args()

	gen = load_network(f'./checkpoint/{args.ckpt}')

	with open(f'./dataset/DATASET_acts.pickle', 'rb') as handle:
		real_acts = pickle.load(handle)

	inception = InceptionV3().cuda()

	args.gen_sample = {512: (8, 4), 1024: (4, 2)}
	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))

	fixed_noise = torch.randn(gen_i, gen_j, code_size)


	metrics = evaluate(Generator=gen, fixed_noise=fixed_noise, inception=inception,
							   real_acts=real_acts, name=args.name)


	for (key, val) in metrics.items():
		print("metrics", key, val)



