import torch
import torch.nn as nn
import snntorch.surrogate as snnfunc


class ExpNeuron(nn.Module):
  	# Реализована модель, где reset_mechanism set to membrane_min
	# После спайка данная модель обнуляет время нейрона и сбрасывает сумму спайков

	# Обдумать момент с подавлением активности нейрона после испускания спайка

	# Если init_hidden = True, список используется для инициализации и обновления состояния нейронов
	instances = []

	def __init__( 
			self, 
			beta = 1e-1, 
			threshold = 1.0, 
			membrane_zero = 0.5, 
			spike_grad = None, 
			surrogate_disable = False, 
			learn_threshold = False, 
			learn_beta = False, 
			learn_membrane_zero = False,
			output = False, 
			# Реализовать механизм подавления других нейронов
			inhibition = False,
			init_hidden = False, 
			# Масштабирование спайков 
			graded_spikes_factor = 1.0, 
			learn_graded_spikes_factor = False):
		
		
		super().__init__()

		ExpNeuron.instances.append(self)

		# Выбор суррагатной функции.
		# atan() - default
		# Если surrogate_disable  
		if surrogate_disable:
			# _surrogate_bypass используется для обхода суррогатного градиента
			self.spike_grad = self._surrogate_bypass
		elif spike_grad == None:
			self.spike_grad = snnfunc.atan()
		else:
			self.spike_grad = spike_grad

		self.init_hidden = init_hidden
		self.output = output
		self.inhibition = inhibition
		self.surrogate_disable = surrogate_disable

		self._snn_register_buffer(
			threshold = threshold,
			beta = beta,
			membrane_zero = membrane_zero,
			learn_membrane_zero = learn_membrane_zero,
			learn_threshold = learn_threshold,
			learn_beta = learn_beta,
			graded_spikes_factor = graded_spikes_factor,
			learn_graded_spikes_factor = learn_graded_spikes_factor)
		self._init_membrane()
		
	def _snn_register_buffer(
		self,
		threshold,
		beta,
		membrane_zero,
		learn_threshold,
		learn_beta,
		learn_membrane_zero,
		graded_spikes_factor,
		learn_graded_spikes_factor):

		self._threshold_buffer(threshold, learn_threshold)
		self._beta_buffer(beta, learn_beta)
		self._membrane_zero_buffer(membrane_zero, learn_membrane_zero)
		self._graded_spikes_buffer(graded_spikes_factor, learn_graded_spikes_factor)

	def _threshold_buffer(self, threshold, learn_threshold):
		if not isinstance(threshold, torch.Tensor):
			threshold = torch.as_tensor(threshold)
		if learn_threshold:
			self.threshold = nn.Parameter(threshold)
		else:
			self.register_buffer("threshold", threshold)

	def _beta_buffer(self, beta, learn_beta):
		if not isinstance(beta, torch.Tensor):
			beta = torch.as_tensor(beta)
		if learn_beta:
			self.beta = nn.Parameter(beta)
		else:
			self.register_buffer("beta", beta)

	def _membrane_zero_buffer(self, membrane_zero, learn_membrane_min):
		if not isinstance(membrane_zero, torch.Tensor):
			membrane_zero = torch.as_tensor(membrane_zero)
		if learn_membrane_min:
			self.membrane_zero = nn.Parameter(membrane_zero)
		else:
			self.register_buffer("membrane_zero", membrane_zero)

	def _graded_spikes_buffer(self, graded_spikes_factor, learn_graded_spikes_factor):
		if not isinstance(graded_spikes_factor, torch.Tensor):
			graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
		if learn_graded_spikes_factor:
			self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
		else:
			self.register_buffer("graded_spikes_factor", graded_spikes_factor)

	def fire(self, membrane):
		membrane_shift = membrane - self.threshold
		spk = self.spike_grad(membrane_shift)

		spk = spk * self.graded_spikes_factor

		return spk
	
	# Подавляет остальные нейроны при испускании спайка нейрона с наибольшим threshold
	def fire_inhibition(self, batch_size, mem):
		pass

	def _membrane_reset(self, membrane):
		# Если surrogate_disable, то reset принимает бинарные значения 

		membrane_shift = membrane - self.threshold
		
		# Осоединение от графа вычислений
		reset = self.spike_grad(membrane_shift).clone().detach()

		return reset

	def _init_membrane(self):
		membrane = torch.zeros(1)
		neuron_time = torch.zeros(1)
		input_sum = torch.zeros(1)
		self.register_buffer("neuron_time", neuron_time, False)
		self.register_buffer("input_sum", input_sum, False)
		self.register_buffer("membrane", membrane, False)

	# Требует доработки, так как необходимо еще и взаимодействовать с буфером, чтобы правильно сбросить потенциал
	def reset_membrane(self):
		self.membrane.zero_()
		self.neuron_time.zero_()
		self.input_sum.zero_()

		return self.membrane_zero
			
	def init_neuron(self):
		return self.reset_membrane()
	
	def forward(self, input, membrane = None):
		if membrane is not None: 
			self.membrane.copy_(membrane)

		if self.init_hidden and membrane != None:
			raise TypeError("Membrane shouldnt be passed while init_hidden is true")

		# Изменение размера для обучения батчами
		if self.membrane.shape != input.shape:
			# Вероятно возникнет ошибка при многомерных данных
			try:
				self.membrane = torch.ones_like(input) * self.membrane.item()
			except Exception as e:
				print("Multiplying membrane tensor error: ", e)

			self.input_sum = torch.zeros_like(input)
			self.neuron_time = torch.zeros_like(input)
			# Умножение тензора на тензор
			try:
				self.membrane_zero = torch.ones_like(input) * self.membrane_zero.item()
			except Exception as e:
				print("Multiplying membrane_zero tensor error: ", e)

		self.reset = self._membrane_reset(self.membrane)
		self.membrane = self._membrane_set_to(input)

		# Реализовать fire_inhibition
		if self.inhibition:
			spk = self.fire_inhibition(self.membrane.size(0), self.membrane)
		else:
			spk = self.fire(self.membrane)

		if self.output:
			return spk, self.membrane
		elif self.init_hidden:
			return spk
		else:
			return spk, self.membrane

	def _membrane_update_function(self, input):

		self.input_sum = self.input_sum + input
		# В данном случае beta служит произведением констант beta, опр. моделью, и time_step size. 
		update = self.membrane_zero * torch.exp(-self.beta*self.neuron_time + self.input_sum)

		# Следующий шаг времени
		self.neuron_time = self.neuron_time + 1	
		
		return update

	def _membrane_set_to(self, input):
		# При reset близком к 1 обнуляется сумма накопленных спайков 
		"""
		Ошибка возникает в этом блоке, почему? Что-то не то с обновлением суммы, хотя раньше все было ок, есть вероятность, что важен контекст применения в программе 
		"""
		self.input_sum = self.input_sum - self.reset * (self.input_sum)

		self.neuron_time = self.neuron_time - self.reset * (self.neuron_time)
		return self._membrane_update_function(input)
		
	@classmethod
	def init(cls):
		cls.instances = []

	# def detach_state_(self):
	# 	if getattr(self, "membrane", None) is not None:
	# 		self.membrane.detach_()
	# 	if getattr(self, "input_sum", None) is not None:
	# 		self.input_sum.detach_()
	# 	if getattr(self, "neuron_time", None) is not None:
	# 		self.neuron_time.detach_()

	# def reset_state_(self):
	# 	self.membrane    = None
	# 	self.input_sum   = None
	# 	self.neuron_time = None

	# membrane_zero / membrane_min
	@classmethod
	def detach_hidden(cls):
		"""Returns the hidden states, detached from the current graph.
		Intended for use in truncated backpropagation through time where
		hidden state variables are instance variables."""

		for layer in range(len(cls.instances)):
			if isinstance(cls.instances[layer], ExpNeuron):
				cls.instances[layer].membrane.detach_()
				cls.instances[layer].input_sum.detach_()
				cls.instances[layer].neuron_time.detach_()


	# membrane_zero / membrane_min
	@classmethod
	def reset_hidden(cls):
		"""Used to clear hidden state variables to zero.
		Intended for use where hidden state variables are instance variables.
		Assumes hidden states have a batch dimension already."""
		for layer in range(len(cls.instances)):
			if isinstance(cls.instances[layer], ExpNeuron):
				cls.instances[layer].membrane = torch.zeros_like(
					cls.instances[layer].membrane,
					device=cls.instances[layer].membrane.device,
				)
				cls.instances[layer].input_sum = torch.zeros_like(
					cls.instances[layer].input_sum,
					device=cls.instances[layer].input_sum.device,
				)
				cls.instances[layer].neuron_time = torch.zeros_like(
					cls.instances[layer].neuron_time,
					device=cls.instances[layer].neuron_time.device,
				)
			
	@staticmethod
	def _surrogate_bypass(input):
		return (input > 0).float()