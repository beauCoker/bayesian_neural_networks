import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from math import pi
from src.util import kldiv_diag_gaussian, reparam, sample_MVG, sample_MVG2


class BBBLayer(nn.Module):
	"""
	Affine layer, places data dependent sparsity prior on activations
	"""
	def __init__(self, dim_in, dim_out, init_weight=None, init_bias=None, **kwargs):
		super(BBBLayer, self).__init__()

		# architecture
		self.dim_in = dim_in
		self.dim_out = dim_out

		# Variational weight (W) and bias (b) parameters
		self.W_mean = nn.Parameter(torch.Tensor(dim_out, dim_in))
		self.b_mean = nn.Parameter(torch.Tensor(dim_out))

		self.W_logvar = nn.Parameter(torch.Tensor(dim_out, dim_in))
		self.b_logvar = nn.Parameter(torch.Tensor(dim_out))

		# Priors
		self.register_buffer('W_var_prior', torch.Tensor([2. / self.dim_in]))
		self.register_buffer('W_mean_prior', torch.Tensor([0.]))
		self.register_buffer('b_var_prior', torch.Tensor([1.]))
		self.register_buffer('b_mean_prior', torch.Tensor([0.]))

		#self.W_var_prior = torch.Tensor([1. / self.dim_in])
		#self.W_mean_prior = torch.Tensor([0.])

		#self.b_var_prior = torch.Tensor([1.])
		#self.b_mean_prior = torch.Tensor([0.])

		# init params either random or with pretrained net
		self.init_parameters(init_weight, init_bias)

	def init_parameters(self, init_weight, init_bias):
		# init means
		if init_weight is not None:
			self.W_mean.data = torch.Tensor(init_weight)
		else:
			#self.W_mean.data.normal_(0, np.float(self.W_var_prior.numpy()[0]))
			self.W_mean.data.normal_(0.0,.05)

		if init_bias is not None:
			self.b_mean.data = torch.Tensor(init_bias)
		else:
			#self.b_mean.data.normal_(0, 1)
			self.b_mean.data.normal_(0.0, .05)

		# init variances
		#self.W_logvar.data.normal_(-9, 1e-2)
		#self.b_logvar.data.normal_(-9, 1e-2)

		self.W_logvar.data.normal_(-5, .05)
		self.b_logvar.data.normal_(-5, .05)


	def sample_posterior(self, n_samp):
		# Samples weights from variational posterior
		normal = Normal(loc=self.W_mean.detach().view(-1), \
						scale=self.W_logvar.detach().exp().sqrt().view(-1))
		return normal.sample((n_samp,))

	def forward(self, x, sample=True):
		# local reparameterization trick

		mu_activations = F.linear(x, self.W_mean, self.b_mean)
		var_activations = F.linear(x.pow(2), self.W_logvar.exp(), self.b_logvar.exp())
		
		activ = reparam(mu_activations, var_activations, sample=sample)
		return activ

	def kl_divergence(self):
		"""
		KL divergence (q(W) || p(W))
		:return:
		"""
		W_kldiv = kldiv_diag_gaussian(self.W_mean, self.W_logvar.exp(), self.W_mean_prior, self.W_var_prior)
		b_kldiv = kldiv_diag_gaussian(self.b_mean, self.b_logvar.exp(), self.b_mean_prior, self.b_var_prior)
		return W_kldiv + b_kldiv


	def kl_divergence_kde(self,n_samp=5):

		eps = torch.randn((n_samp, self.W_mean.view(-1).shape[0]))

		Wq = self.W_mean.view(1,-1).repeat(n_samp,1) + self.W_logvar.exp().sqrt().view(1,-1).repeat(n_samp,1)*eps


		normal = Normal(loc=self.W_mean_prior.view(-1), \
						scale=self.W_var_prior.sqrt().view(-1))
		Wp = normal.sample((n_samp, Wq.shape[1]))

		#print("Wq",Wq.requires_grad)
		#print("Wp",Wp.requires_grad)

		n = Wq.shape[0]
		m = Wp.shape[0]

		d = Wq.shape[1]
		wp_dist, _ = torch.min(torch.sqrt(torch.sum((Wq.unsqueeze(1) - Wp)**2,2)+1e-8), 1)

		ww_dist = torch.sqrt(torch.sum((Wq.unsqueeze(1) - Wq)**2,2) +1e-8)
		ww_dist,_ = torch.min(ww_dist,1)

		kl_W = (d/n)*torch.sum(torch.log(wp_dist / (ww_dist+1e-8) + 1e-8) + torch.log(torch.tensor(m / (n-1))))

		return kl_W


class MVGLayer(nn.Module):
	"""
	"""
	def __init__(self, dim_in, dim_out, **kwargs):
		super(MVGLayer, self).__init__()

		# architecture
		self.dim_in = dim_in + 1
		self.dim_out = dim_out

		# Variational weight (W) and bias (b) parameters
		self.W_mean = nn.Parameter(torch.Tensor(self.dim_out, self.dim_in))

		self.W_in_logvar = nn.Parameter(torch.Tensor(self.dim_in)) # U
		self.W_out_logvar = nn.Parameter(torch.Tensor(self.dim_out)) # V

		# Priors
		#self.W_mean_prior = torch.Tensor([0.])
		#self.W_var_in_prior = torch.Tensor([1. / self.dim_in])
		#self.W_var_out_prior = torch.Tensor([1. / self.dim_out])

		self.init_parameters()

	def init_parameters(self):
		# init means
		self.W_mean.data.normal_(0, .05)

		# init variances
		self.W_in_logvar.data.normal_(-5, .05)
		self.W_out_logvar.data.normal_(-5, .05)

	def sample_posterior(self, n_samp):
		return torch.stack([sample_MVG(self.W_mean.transpose(0,1), torch.diag(self.W_in_logvar.exp()), self.W_out_logvar.exp()).view(-1) for _ in range(n_samp)])


	def forward(self, x, sample=True):
		# local reparameterization trick

		x = torch.cat((torch.ones(x.shape[0],1),x),dim=1) # Add column of ones since no bias
		
		mu_activations = F.linear(x, self.W_mean)
		if not sample:
			active = mu_activations
		else:
			u = self.W_in_logvar.exp() # Diagonal of U
			v = self.W_out_logvar.exp() # Diagonal of V

			var_in_activations = torch.mm(torch.mm(x,torch.diag(u)), x.transpose(0,1))
			var_out_activations = v
			active = sample_MVG(mu_activations, var_in_activations, var_out_activations)
			#active = sample_MVG2(mu_activations, var_in_activations, torch.diag(var_out_activations))
		return active

	def kl_divergence(self):
		"""
		KL divergence (q(W) || p(W))
		:return:
		"""
		# V: columns, out
		# U: rows, in

		return 0.5 * ( \
			torch.sum(self.W_in_logvar.exp())*torch.sum(self.W_out_logvar.exp()) \
			+ torch.sum(self.W_mean**2) \
			- self.dim_in*self.dim_out \
			- self.dim_out*torch.sum(self.W_in_logvar) \
			- self.dim_in*torch.sum(self.W_out_logvar)
		)


class MNFLayer(nn.Module):
	def __init__(self, dim_in, dim_out, n_flows_q=2, n_flows_r=2, flow_dim_h=10, **kwargs):
		super(MNFLayer, self).__init__()


		self.dim_in = dim_in
		self.dim_out = dim_out

		self.bernoulli = torch.distributions.Bernoulli(probs=.5)

		# Variational weight (W) and bias (b) parameters
		self.W_mean = nn.Parameter(torch.Tensor(dim_out, dim_in))
		self.W_logvar = nn.Parameter(torch.Tensor(dim_out, dim_in))

		self.b_mean = nn.Parameter(torch.Tensor(dim_out))
		self.b_logvar = nn.Parameter(torch.Tensor(dim_out))

		# Parameters of q distribution
		self.q_mean = nn.Parameter(torch.Tensor(dim_in))
		self.q_logvar = nn.Parameter(torch.Tensor(dim_in))

		# q normalizing flow
		self.f_q = nn.ModuleList([nn.Linear(dim_in, flow_dim_h) for flow in range(n_flows_q)])
		self.g_q = nn.ModuleList([nn.Linear(flow_dim_h, dim_in) for flow in range(n_flows_q)])
		self.k_q = nn.ModuleList([nn.Linear(flow_dim_h, dim_in) for flow in range(n_flows_q)])

		# Parameters of r distribution
		self.c = nn.Parameter(torch.Tensor(dim_in))
		self.b1 = nn.Parameter(torch.Tensor(dim_in))
		self.b2 = nn.Parameter(torch.Tensor(dim_in))

		self.f_r = nn.ModuleList([nn.Linear(dim_in, flow_dim_h) for flow in range(n_flows_r)])
		self.g_r = nn.ModuleList([nn.Linear(flow_dim_h, dim_in) for flow in range(n_flows_r)])
		self.k_r = nn.ModuleList([nn.Linear(flow_dim_h, dim_in) for flow in range(n_flows_r)])

		# Priors
		self.W_mean_prior = torch.Tensor([0.])
		self.W_var_prior = torch.Tensor([2. / self.dim_in])

		self.b_mean_prior = torch.Tensor([0.])
		self.b_var_prior = torch.Tensor([1.])

		self.init_varparams()

	def init_varparams(self):
		# Variational weight (W) and bias (b) parameters
		self.W_mean.data.normal_(0, 1)
		self.W_logvar.data.normal_(-5, .05)

		self.b_mean.data.normal_(0, 1)
		self.b_logvar.data.normal_(-5, .05)

		# Parameters of q distribution
		self.q_mean.data.normal_(0, .05)
		self.q_logvar.data.normal_(-5, .05)

		# Parameters of r distribution
		self.c.data.normal_(0, .05)
		self.b1.data.normal_(0, .05)
		self.b2.data.normal_(0, .05)


	def norm_flow(self, z, f, g, k, sample=True):
		logdets = torch.zeros(z.shape[0])

		for flow in range(len(f)):

			# Sample the mask
			m = self.bernoulli.sample(z.shape) if sample else 0.5

			h = torch.tanh(f[flow](m * z))

			mu = g[flow](h)
			sigma = torch.sigmoid(k[flow](h))

			z = m*z + (1-m)*(z*sigma+(1-sigma)*mu)
			logdets += torch.sum((1-m)*torch.log(sigma), dim=1)

		return z, logdets


	def sample_q(self, n_samp, sample=True):
		'Sample z0 ~ q()'
		if sample:
			eps = torch.randn((n_samp,self.dim_in))
			return self.q_mean + torch.sqrt(torch.exp(self.q_logvar)) * eps
		else:
			return self.q_mean.repeat((n_samp,1))

	def sample_z(self, n_samp=1, sample=True):
		'Sample z_T from Eq. 6'

		# Sample from q
		Z_0 = self.sample_q(n_samp=n_samp, sample=sample)

		# Apply normalizing flow
		Z_T, logdets = self.norm_flow(Z_0, self.f_q, self.g_q, self.k_q, sample=sample)

		return Z_T, logdets

	def kl_divergence(self):
		'KL divergence as in eq. 13'

		# Sample z_Tf
		z_Tf, logdet = self.sample_z()

		# Conditonal distribution of weights q(W | z) eq. 4
		Wz_mean = z_Tf * self.W_mean
		Wz_logvar = self.W_logvar

		## KL term
		kldiv_W = kldiv_diag_gaussian(Wz_mean, torch.exp(Wz_logvar), self.W_mean_prior, self.W_var_prior)
		kldiv_b = kldiv_diag_gaussian(self.b_mean, torch.exp(self.b_logvar), self.b_mean_prior, self.b_var_prior)
		kldiv = kldiv_W + kldiv_b

		## r term -- eq. 15
		z_Tb, logdet_r = self.norm_flow(z_Tf, self.f_r, self.g_r, self.k_r)

		a = torch.tanh(torch.mm(self.c.view(1,-1),Wz_mean.transpose(0,1))) # (1,dim_out)

		mu_tilde = torch.sum(torch.mm(self.b1.view(-1,1), a)) # Divide by D_out?
		logsig2_tilde = torch.sum(torch.mm(self.b2.view(-1,1), a)) # Divide by D_out?. In code this is -logsig2_tilde
		logr_zTb_W = -.5*torch.sum(torch.log(torch.tensor(2*pi)) + logsig2_tilde + (z_Tb - mu_tilde)**2 / torch.exp(logsig2_tilde)) # eq. 8

		logr_zTf_W = logr_zTb_W + logdet_r[0]

		## q term -- eq. 16
		logq_z0  = -.5*torch.sum(torch.log(torch.tensor(2*pi)) + self.q_logvar + 1) 
		logq_zTf = logq_z0 - logdet[0]

		return kldiv - logr_zTf_W + logq_zTf 
		#return kldiv + logq_zTf 

	def sample_posterior(self, n_samp):
		# Samples weight from variational posterior
		W_samps = []
		for i in range(n_samp):

			z_Tf, _ = self.sample_z()

			# Conditonal distribution of weights q(W | z) eq. 4
			Wz_mean = z_Tf * self.W_mean
			Wz_logvar = self.W_logvar

			normal = Normal(loc=Wz_mean, scale=Wz_logvar.exp().sqrt())
			W_samps.append(normal.sample().view(-1))

		return torch.stack(W_samps)

	def forward(self, X, sample=True):
		# Follows Algorithm 1
		# x: (n_obs,input_dim)

		# Mw = self.W_mean (dim_out, dim_in)
		# Sigma_w = self.b_mean (dim_out, dim_in)

		Z_Tf, _ = self.sample_z(X.shape[0], sample=sample)

		Mh = F.linear(X * Z_Tf, self.W_mean, self.b_mean)
		
		if not sample:
			return Mh
		else:
			W_var = torch.clamp(self.W_logvar.exp(),0,1)
			b_var = torch.clamp(self.b_logvar.exp(),0,1)
			#W_var = self.W_logvar.exp()
			#b_var = self.b_logvar.exp()

			Vh = F.linear(X.pow(2), W_var, b_var)
			E = torch.randn(Vh.shape)
			return Mh + torch.sqrt(Vh) * E

class BBHLayer(nn.Module):
	"""
	"""
	def __init__(self, dim_in, dim_out, dim_z=1, arch_G=[16,32], **kwargs):
		super(BBHLayer, self).__init__()

		self.dim_in = dim_in
		self.dim_out = dim_out
		self.dim_z = dim_z # d

		#self.activation_G = torch.tanh
		self.activation_G = F.relu
		#self.activation_G = lambda x: torch.max(.1*x, x)

		self.G_W = self.build_G(dim_z, arch_G, dim_in * dim_out)
		self.G_b = self.build_G(dim_z, arch_G, dim_out)

		self.normal = torch.distributions.Normal(0, 1/self.dim_in)
		#self.normal = torch.distributions.Normal(0, 1.)


	def build_G(self, dim_z, arch_G, dim_flat):

		arch = [dim_z] + arch_G + [dim_flat]
		return nn.ModuleList([nn.Linear(arch[i], arch[i+1]) for i in range(len(arch)-1)])


	def sample_from_G(self, n_samp, G):
		z = torch.randn(n_samp, self.dim_z)

		for i in range(len(G)-1):
			z = self.activation_G(G[i](z))
		return G[-1](z)

	def kl_divergence(self,n_samp=5):

		Wq = self.sample_from_G(n_samp, self.G_W)
		Wp = self.normal.sample((n_samp, Wq.shape[1]))


		n = Wq.shape[0]
		m = Wp.shape[0]

		d = Wq.shape[1]
		wp_dist, _ = torch.min(torch.sqrt(torch.sum((Wq.unsqueeze(1) - Wp)**2,2)+1e-8), 1)

		ww_dist = torch.sqrt(torch.sum((Wq.unsqueeze(1) - Wq)**2,2) +1e-8)
		ww_dist,_ = torch.min(ww_dist,1)

		kl_W = (d/n)*torch.sum(torch.log(wp_dist / (ww_dist+1e-8) + 1e-8) + torch.log(torch.tensor(m / (n-1))))

		Wqb = self.sample_from_G(n_samp, self.G_b)
		Wpb = torch.randn((n_samp, Wqb.shape[1]))

		nb = Wqb.shape[0]
		mb = Wpb.shape[0]

		db = Wqb.shape[1]
		wp_distb, _ = torch.min(torch.sqrt(torch.sum((Wqb.unsqueeze(1) - Wpb)**2,2)+1e-8), 1)

		ww_distb = torch.sqrt(torch.sum((Wqb.unsqueeze(1) - Wqb)**2,2) +1e-8)
		ww_distb,_ = torch.min(ww_distb,1)

		kl_b = (db/nb)*torch.sum(torch.log(wp_distb / (ww_distb+1e-8) + 1e-8) + torch.log(torch.tensor(mb / (nb-1))))

		return kl_W + kl_b


	def kl_divergence_indep(self, n_samp=5):

		Wq = self.sample_from_G(n_samp, self.G_W).view(-1)
		#Wp = torch.randn(Wq.shape).view(-1)
		Wp = self.normal.sample(Wq.shape).view(-1)
		#Wp = torch.randn(np.prod(Wq.shape)*2).view(-1)
		#Wp = torch.zeros(Wq.shape)

		n = Wq.shape[0]
		m = Wp.shape[0]

		wp_dist,_ = torch.min(torch.sqrt((Wq.view(-1,1) - Wp.view(1,-1))**2 + 1e-8),1)

		ww_dist = torch.sqrt((Wq.view(-1,1) - Wq.view(1,-1))**2 + 1e-8) + 1e10*torch.eye(n,n)
		ww_dist,_ = torch.min(ww_dist,1)


		kl_W = (1/n)*torch.sum(torch.log(wp_dist / (ww_dist+1e-8) + 1e-8) + torch.log(torch.tensor(m / (n-1))))

		## b

		Wqb = self.sample_from_G(n_samp, self.G_b).view(-1)
		Wpb = torch.randn(Wqb.shape).view(-1)

		nb = Wqb.shape[0]
		mb = Wpb.shape[0]

		wp_distb,_ = torch.min(torch.sqrt((Wqb.view(-1,1) - Wpb.view(1,-1))**2 + 1e-8),1)

		ww_distb = torch.sqrt((Wqb.view(-1,1) - Wqb.view(1,-1))**2 + 1e-8) + 1e10*torch.eye(nb,nb)
		ww_distb,_ = torch.min(ww_distb,1)

		kl_b = (1/n)*torch.sum(torch.log(wp_distb / (ww_distb+1e-8) + 1e-8) + torch.log(torch.tensor(mb / (nb-1))))

		return kl_W + kl_b
		


	def sample_posterior(self, n_samp):
		return self.sample_from_G(n_samp, self.G_W)

	def forward(self, x, sample=True):
		W = self.sample_from_G(1, self.G_W).view(self.dim_out, self.dim_in)
		b = self.sample_from_G(1, self.G_b).view(self.dim_out)

		return F.linear(x,W,b)




















