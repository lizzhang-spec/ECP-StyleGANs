import argparse
import datetime
import math
import pickle
import random

import numpy as np
from torch import nn, optim
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from tqdm import tqdm

from GA import *
from dataset import MultiResolutionDataset
from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid
from model import *
from train import accumulate, sample_data, adjust_lr
from utils import compute_layer_mask
from model import EqualConv2d

device = torch.device('cuda')


def requires_grad(model, flag=True, target_layer=None):
	for name, param in model.named_parameters():
		if target_layer is None: 
			param.requires_grad = flag
		elif target_layer in name:  
			param.requires_grad = flag


def evaluate(iteration, G_running_target, fixed_noise, inception, real_acts, fid_values, current_fitness):
	"""Custom GAN evaluation function"""


	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))

	images = []
	with torch.no_grad():
		for i in range(gen_i):
			images.append(G_running_target(fixed_noise[i].cuda(), step=step, alpha=alpha).cpu())

	sample_path = f'sample/{args.name}/{str(iteration).zfill(6)}.png'
	utils.save_image(torch.cat(images, dim=0), sample_path, nrow=gen_i, normalize=True, value_range=(-1, 1))


	sample_num = args.sample_num
	fake_images, fake_acts = get_fake_images_and_acts(inception, G_running_target, code_size, step, alpha, sample_num, batch_size)

	fid = compute_fid(real_acts, fake_acts)
	fid_values.append(fid)

	if len(fid_values) > 1:  
		mean_fid = np.mean(fid_values)
		std_fid = np.std(fid_values)
		scaled_fid = scale_fid_with_sigmoid(fid, mean=mean_fid, std=std_fid, normalize=True)
	else:
		scaled_fid = scale_fid_with_sigmoid(fid, normalize=False)

	mask_best = mask_all[np.argmin(current_fitness)]
	mask_best_full = compute_layer_mask(mask_best, mask_chns)
	mask_best_full = [y for x in mask_best_full for y in x]
	mask_best_full = np.array(mask_best_full)
	scaled_CH = float((sum(mask_best_full)) / len(mask_best_full))

	metrics = {'scaled_fid': scaled_fid, 'scaled_CH': scaled_CH}

	return metrics


def l2_reg(net_src, net_tgt):
	params_src = list(net_src.parameters())
	params_tgt = list(net_tgt.parameters())

	loss = 0
	for p_src, p_tgt in zip(params_src, params_tgt):
		loss += F.mse_loss(p_tgt, p_src)

	return loss


def sample_noise(batch_size):
	if args.mixing and random.random() < 0.9:
		gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, batch_size, code_size, device='cuda').chunk(4, 0)
		gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
		gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

	else:
		gen_in1, gen_in2 = torch.randn(2, batch_size, code_size, device='cuda').chunk(2, 0)
		gen_in1 = gen_in1.squeeze(0)
		gen_in2 = gen_in2.squeeze(0)

	return gen_in1, gen_in2


def FM_reg(real_image, feature_loc, D_source, D_target):
	feat_src = D_source(real_image, step=step, alpha=alpha, get_feature=True, feature_loc=feature_loc)
	feat_tgt = D_target(real_image, step=step, alpha=alpha, get_feature=True, feature_loc=feature_loc)

	return F.mse_loss(feat_tgt, feat_src)


def backward_D(args, G_target, D_target, real_image, gen_in, D_source):
	real_image = real_image.cuda()

	real_image.requires_grad = True

	real_predict = D_target(real_image, step=step, alpha=alpha)
	D_loss_real = F.softplus(-real_predict).mean()

	grad_real = grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
	grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
	grad_penalty = 10 / 2 * grad_penalty

	fake_image_tgt = G_target(gen_in, step=step, alpha=alpha)
	fake_predict = D_target(fake_image_tgt, step=step, alpha=alpha)
	D_loss_fake = F.softplus(fake_predict).mean()


	if args.lambda_FM > 0:
		FM_loss = FM_reg(real_image, args.feature_loc, D_source, D_target) * args.lambda_FM
	else:
		FM_loss = 0

	if args.lambda_l2_D > 0:
		l2_D_loss = l2_reg(D_source, D_target) * args.lambda_l2_D
	else:
		l2_D_loss = 0

	(D_loss_real + D_loss_fake + grad_penalty + FM_loss + l2_D_loss).backward()

	D_loss_val = (D_loss_real + D_loss_fake).item()
	grad_loss_val = grad_penalty.item() if grad_penalty > 0 else 0

	return D_loss_val, grad_loss_val


def backward_G(args, G_target, D_target, gen_in)：

	fake_image_tgt = G_target(gen_in, step=step, alpha=alpha)
	predict = D_target(fake_image_tgt, step=step, alpha=alpha)
	gen_loss = F.softplus(-predict).mean()


	l2_G_loss = 0

	(gen_loss + l2_G_loss).backward()

	G_loss_val = gen_loss.item()

	return G_loss_val


def scale_fid_with_sigmoid(fid, mean=0, std=1, normalize=False):
	if normalize:
		fid = (fid - mean) / std 
	scaled_fid = 1 / (1 + np.exp(-fid))  
	return scaled_fid


def finetune(args, dataset, G_target, D_target, G_optimizer, D_optimizer, G_running_target, D_source,
			 fixed_noise, inception, real_acts, fitness_id, current_fitness, generation):

	if not os.path.exists(f'checkpoint/{args.name}/{generation}/{fitness_id}'):
		os.makedirs(f'checkpoint/{args.name}/{generation}/{fitness_id}')

	if not os.path.exists(f'sample/{args.name}/{generation}/{fitness_id}'):
		os.makedirs(f'sample/{args.name}/{generation}/{fitness_id}')

	logger = SummaryWriter(f'checkpoint/{args.name}/{generation}/{fitness_id}')


	global step 
	global batch_size  

	step = int(math.log2(args.image_size)) - 2
	resolution = 4 * 2 ** step

	batch_size = args.batch.get(resolution, args.batch_default) * torch.cuda.device_count()  
	loader = sample_data(dataset, batch_size, resolution)
	loader_iter = iter(loader)


	fid_values = []  

	pbar = tqdm(range(args.phase), position=0)

	metrics = evaluate(iteration=0, G_running_target=G_running_target, fixed_noise=fixed_noise, inception=inception,
					   real_acts=real_acts, fid_values = fid_values, current_fitness=current_fitness)
	for key, val in metrics.items():
		logger.add_scalar(key, val, 0)

	best_fitness = metrics['scaled_fid'] + metrics['scaled_CH']


	for i in pbar:
		adjust_lr(G_optimizer, args.lr.get(resolution, 0.001))
		adjust_lr(D_optimizer, args.lr.get(resolution, 0.001))


		try:
			real_index, real_image = next(loader_iter)

		except:
			loader_iter = iter(loader)
			real_index, real_image = next(loader_iter)

		gen_in1, gen_in2 = sample_noise(len(real_image))


		D_target.zero_grad()

		requires_grad(G_target, False)
		if args.freeze_D:
			for loc in range(args.feature_loc):
				requires_grad(D_target, True, target_layer=f'progression.{8 - loc}')
			requires_grad(D_target, True, target_layer=f'linear')
		else:
			requires_grad(D_target, True)

		D_loss_val, grad_loss_val = backward_D(args, G_target, D_target, real_image, gen_in1, D_source)

		D_optimizer.step()


		G_target.zero_grad()

		requires_grad(G_target, True)  
		if args.freeze_D:
			for loc in range(args.feature_loc):
				requires_grad(D_target, False, target_layer=f'progression.{8 - loc}')
			requires_grad(D_target, False, target_layer=f'linear')
		else:
			requires_grad(D_target, False)

		G_loss_val = backward_G(args, G_target, D_target, gen_in2)

		G_optimizer.step()

		accumulate(G_running_target, G_target)


		if (i + 1) % args.eval_step == 0:
			logger.add_scalar('G_loss_val', G_loss_val, i + 1)
			logger.add_scalar('D_loss_val', D_loss_val, i + 1)
			logger.add_scalar('grad_loss_val', grad_loss_val, i + 1)

			metrics = evaluate(iteration=i + 1, G_running_target=G_running_target, fixed_noise=fixed_noise,
							   inception=inception, real_acts=real_acts, fid_values = fid_values, current_fitness=current_fitness)
			for key, val in metrics.items():
				logger.add_scalar(key, val, i + 1)

			if metrics['scaled_fid'] + metrics['scaled_CH'] < best_fitness:
				current_fitness[fitness_id] = metrics['scaled_fid'] + metrics['scaled_CH']
				torch.save({
					'generator': G_target.state_dict(),
					'discriminator': D_target.state_dict(),
					'g_optimizer': G_optimizer.state_dict(),
					'd_optimizer': D_optimizer.state_dict(),
					'g_running': G_running_target.state_dict(),
				}, f'checkpoint/{args.name}/{generation}/{fitness_id}/best.model')

		state_msg = f'Size: {4 * 2 ** step}; G: {G_loss_val:.3f}; D: {D_loss_val:.3f}; Grad: {grad_loss_val:.3f};'
		if metrics is not None:
			state_msg += ''.join([f' {key}: {val:.2f};' for (key, val) in metrics.items()])

		pbar.set_description(state_msg)


def caculate_fitness(mask_input, fitness_id, current_fitness, generation):

	args.gen_sample = {512: (8, 4), 1024: (4, 2)}
	args.batch_default = 8

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)


	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
	])

	dataset = MultiResolutionDataset(f'./dataset/{args.dataset}_lmdb', transform, resolution=args.image_size)

	G_target = nn.DataParallel(StyledGenerator(code_size)).cuda()
	D_target = nn.DataParallel(Discriminator(from_rgb_activate=True)).cuda()
	G_running_target = StyledGenerator(code_size).cuda()
	G_running_target.train(False)
	accumulate(G_running_target, G_target.module, 0)

	D_source = nn.DataParallel(Discriminator(from_rgb_activate=True)).cuda()
	requires_grad(D_source, False)

	if args.freeze_D:
		requires_grad(D_target, False) 

	G_optimizer = optim.Adam(G_target.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
	G_optimizer.add_param_group({'params': G_target.module.style.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01})

	D_optimizer = optim.Adam(D_target.parameters(), lr=args.lr, betas=(0.0, 0.99))

	ckpt = torch.load(args.ckpt)

	if not args.init_G:
		G_target.module.load_state_dict(ckpt['generator'], strict=False)
		G_running_target.load_state_dict(ckpt['g_running'], strict=False)

	if not args.init_D:
		D_target.module.load_state_dict(ckpt['discriminator'])

	if args.sched:
		args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
		args.batch = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 8, 256: 8}
	else:
		args.lr = {}
		args.batch = {}

	D_source.module.load_state_dict(ckpt['discriminator'])

	fitness = 0
	cfg_mask = compute_layer_mask(mask_input, mask_chns)
	cfg_full_mask = [y for x in cfg_mask for y in x]
	cfg_full_mask = np.array(cfg_full_mask)

	
	cfg_id = 0
	start_mask = cfg_mask[cfg_id]

	for m_1 in G_target.modules():
		if isinstance(m_1, StyledConvBlock):

			mask_1 = np.ones(m_1.conv2.conv.weight_orig.data.shape)
			mask_bias_1 = np.ones(m_1.conv2.conv.bias.data.shape)

			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))

			mask_1[:, idx0.tolist(), :, :] = 0
			mask_1[idx0.tolist(), :, :, :] = 0
			mask_bias_1[idx0.tolist()] = 0

			m_1.conv2.conv.weight_orig.data = m_1.conv2.conv.weight_orig.data * torch.FloatTensor(mask_1).cuda()
			m_1.conv2.conv.bias.data = m_1.conv2.conv.bias.data * torch.FloatTensor(mask_bias_1).cuda()

			m_1.conv2.conv.weight_orig.data[:, idx0.tolist(), :, :].requires_grad = False
			m_1.conv2.conv.weight_orig.data[idx0.tolist(), :, :, :].requires_grad = False
			m_1.conv2.conv.bias.data[idx0.tolist()].requires_grad = False

			cfg_id += 1
			if cfg_id < len(cfg_mask):
				start_mask = cfg_mask[cfg_id]
			continue

	cfg_id = 0
	start_mask = cfg_mask[cfg_id]

	for m_2 in G_running_target.modules():
		if isinstance(m_2, StyledConvBlock):

			mask_2 = np.ones(m_2.conv2.conv.weight_orig.data.shape)
			mask_bias_2 = np.ones(m_2.conv2.conv.bias.data.shape)

			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))

			mask_2[:, idx0.tolist(), :, :] = 0
			mask_2[idx0.tolist(), :, :, :] = 0
			mask_bias_2[idx0.tolist()] = 0

			m_2.conv2.conv.weight_orig.data = m_2.conv2.conv.weight_orig.data * torch.FloatTensor(mask_2).cuda()
			m_2.conv2.conv.bias.data = m_2.conv2.conv.bias.data * torch.FloatTensor(mask_bias_2).cuda()

			m_2.conv2.conv.weight_orig.data[:, idx0.tolist(), :, :].requires_grad = False
			m_2.conv2.conv.weight_orig.data[idx0.tolist(), :, :, :].requires_grad = False
			m_2.conv2.conv.bias.data[idx0.tolist()].requires_grad = False

			cfg_id += 1
			if cfg_id < len(cfg_mask):
				start_mask = cfg_mask[cfg_id]
			continue


	inception = nn.DataParallel(InceptionV3()).cuda()

	gen_i, gen_j = args.gen_sample.get(args.image_size, (10, 5))
	fixed_noise = torch.randn(gen_i, gen_j, code_size)

	real_images = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
	with open(f'./dataset/{args.dataset}_acts.pickle', 'rb') as handle:
		real_acts = pickle.load(handle)


	finetune(args, dataset, G_target, D_target, G_optimizer, D_optimizer, G_running_target, D_source,
			 fixed_noise, inception, real_acts, fitness_id, current_fitness, generation)


if __name__ == '__main__':

	code_size = 512
	alpha = 1

	parser = argparse.ArgumentParser(description='Search best mask for StyleGAN Compression(CH_sigFID)')
	parser.add_argument('--dataset', type=str, required=True, help='dataset name')
	parser.add_argument('--name', type=str, default='temp', help='name of experiment')
	parser.add_argument('--ckpt', type=str, default='./checkpoint/stylegan-256px-new.model', help='source model')
	parser.add_argument('--seed', type=int, default=0, help='random seed')

	parser.add_argument('--image_size', default=512, type=int, help='image size')
	parser.add_argument('--phase', type=int, default=250, help='number of samples used for each training phases')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--sched', action='store_true', help='use lr scheduling')
	parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
	parser.add_argument('--loss', type=str, default='r1', choices=['r1'], help='class of gan loss')
	parser.add_argument('--eval_step', default=5, type=int, help='step size for evaluation')
	parser.add_argument('--save_step', default=50, type=int, help='step size for save models')
	parser.add_argument('--sample_num', default=250, type=int, help='number of samples for evaluation')

	parser.add_argument('--init_G', action='store_true', help='initialize G')
	parser.add_argument('--init_D', action='store_true', help='initialize D')
	parser.add_argument('--only_adain', action='store_true', help='only optimize AdaIN layers')
	parser.add_argument('--lambda_l2_G', type=float, default=0, help='weight for l2 loss for G')
	parser.add_argument('--lambda_l2_D', type=float, default=0, help='weight for l2 loss for D')
	parser.add_argument('--lambda_FM', type=float, default=0, help='weight for FM loss for D')
	parser.add_argument('--feature_loc', type=int, default=3, help='feature location for discriminator (default: 3)')
	parser.add_argument('--freeze_D', action='store_true', help='freeze layers of discriminator D')

	args = parser.parse_args()

	max_generation = 25
	population = 16
	s1 = 0.2  # prob for selection
	s2 = 0.7  # prob for crossover
	s3 = 0.1  # prob for mutation
	current_fitness_base = range(population)
	current_fitness = np.asarray(current_fitness_base, dtype=np.float32)

	first_conv_out = 512

	# Define the scaling factors for each convolution layer
	scaling_factors = [1] * 4 + [0.5] * 1 + [0.25] * 1 + [0.125] * 1 + [0.0625] * 1 + [0.03125] * 1
	
	# # Use a list comprehension to compute the channel numbers
	mask_chns = [int(first_conv_out * scale) for scale in scaling_factors]

	bit_len = 0
	for mask_chn in mask_chns:
		bit_len += mask_chn

	print("A new start training")
	if os.path.exists('./log/GA/search') == False:
		os.makedirs('./log/GA/search')

	# initiate population
	mask_all = []
	for i in range(population): 
		mask_all.append(np.random.randint(2, size=bit_len))

	generation = 0

	starttime_1 = datetime.datetime.now()
	for i in range(population):
		print("population = ", i)
		caculate_fitness(mask_all[i], i, current_fitness, generation)
		args.lr = 0.001
		if args.sched:
			args.batch = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 8, 256: 8}
		else:
			args.batch = {}

	endtime_1 = datetime.datetime.now()
	print("The pre-evaluation time is:")
	print((endtime_1 - starttime_1).seconds)

	mask_best = mask_all[np.argmin(current_fitness)]

	np.savetxt('./best_fitness_for_the_first_time.txt', mask_best, delimiter=',')

	best_fitness = min(current_fitness)
	ave_fitness = np.mean(current_fitness)
	print('The best individual for the first time is: %d' % (np.argmin(current_fitness)))
	print('The best fitness for the first time is: %4f' % (best_fitness))

	mask_best_full = compute_layer_mask(mask_best, mask_chns)
	mask_best_full = [y for x in mask_best_full for y in x]
	mask_best_full = np.array(mask_best_full)
	print('The best model channel num for the first time is:%d' % (sum(mask_best_full)))
	print('The ave fitness for the first time is: %4f' % (ave_fitness))
	np.savetxt('./log/GA/search/pre_evaluation_%d_th.txt' % (0), mask_best)


	for j in range(1, max_generation):
		mask_all_current = []

		rest_population = population
		mask_all_current.append(mask_best)
		rest_population -= 1

		while (rest_population > 0):

			s = np.random.uniform(0, 1)
			if s < s1:
				mask_, _ = roulette(mask_all, population, current_fitness)
				mask_all_current.append(mask_)
				rest_population -= 1

			# cross over
			elif (s > s1) & (s <= s1 + s2):
				mask1, mask2 = crossover(mask_all, population, current_fitness, bit_len)

				if rest_population <= 1:
					mask_all_current.append(mask1)
					rest_population -= 1

				else:
					mask_all_current.append(mask1)
					mask_all_current.append(mask2)
					rest_population -= 2

			# mutation
			else:
				mask_ = mutation(mask_all, population, current_fitness, bit_len)
				mask_all_current.append(mask_)
				rest_population -= 1

		mask_all = mask_all_current

		starttime_2 = datetime.datetime.now()

		for i in range(population):
			caculate_fitness(mask_all[i], i, current_fitness, j)
			args.lr = 0.001
			if args.sched:
				args.batch = {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 8, 256: 8}
			else:
				args.batch = {}

		endtime_2 = datetime.datetime.now()
		print("(afhq_search) The evaluation time is:")
		print((endtime_2 - starttime_2).seconds)

		mask_best = mask_all[np.argmin(current_fitness)]

		np.savetxt('./best_fitness.txt', mask_best, delimiter=',')

		best_fitness = min(current_fitness)
		print('The best individual is: %d' % (np.argmin(current_fitness)))
		print('The %d th best fitness is: %4f' % (j, best_fitness))

		mask_best_full = compute_layer_mask(mask_best, mask_chns)
		mask_best_full = [y for x in mask_best_full for y in x]
		mask_best_full = np.array(mask_best_full)
		print('The best model channel num is:%d' % (sum(mask_best_full)))

		ave_fitness = np.mean(current_fitness)
		print('The %d th ave fitness is: %4f' % (j, ave_fitness))

		np.savetxt('./log/GA/search/_%d_th.txt' % (j), mask_best)
