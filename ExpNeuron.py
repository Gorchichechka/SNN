import torch
import torch.nn as nn
import snntorch.surrogate as snnfunc


class ExpNeuron(nn.Module):
  	# Реализована модель, где reset_mechanism set to membrane_min
	# Для упрощения вычислений в данной модели накапливается сумма спайков за время neuron_time
	# После спайка данная модель обнуляет время нейрона и сбрасывает сумму спайков

	# Обдумать момент с подавлением активности нейрона после испускания спайка

	# Если init_hidden = True, список используется для инициализации и обновления состояния нейронов
	instances = []

	def __init__( 
			self, 
			beta = 1e-1, 
			threshold = 1.0, 
			membrane_zero = 0.5, 
			membrane_min = 0.0, 
			spike_grad = None, 
			surrogate_disable = False, 
			learn_threshold = False, 
			learn_beta = False, 
			learn_membrane_min = False, 
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

		# Инициализация внутреннего времени нейрона
		if not isinstance(membrane_zero, torch.Tensor):
			membrane_zero = torch.as_tensor(membrane_zero)
		self.membrane_zero = membrane_zero

		# Инициализация временного шага
		self.dt = 1

		self._snn_register_buffer(
			threshold = threshold,
			beta = beta,
			# membrane_zero = membrane_zero,
			membrane_min= membrane_min,
			learn_threshold = learn_threshold,
			learn_beta = learn_beta,
			# learn_membrane_zero = learn_membrane_zero,
			learn_membrane_min= learn_membrane_min,
			graded_spikes_factor = graded_spikes_factor,
			learn_graded_spikes_factor = learn_graded_spikes_factor)
		# Возможна передача времени и суммы через параметры
		self._init_membrane()
		
	def _snn_register_buffer(
		self,
		threshold,
		beta,
		membrane_min,
		learn_threshold,
		learn_beta,
		learn_membrane_min,
		graded_spikes_factor,
		learn_graded_spikes_factor):

		self._threshold_buffer(threshold, learn_threshold)
		self._beta_buffer(beta, learn_beta)
		self._membrane_min_buffer(membrane_min, learn_membrane_min)
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

	def _membrane_min_buffer(self, membrane_min, learn_membrane_min):
		if not isinstance(membrane_min, torch.Tensor):
			membrane_min = torch.as_tensor(membrane_min)
		if learn_membrane_min:
			self.membrane_min = nn.Parameter(membrane_min)
		else:
			self.register_buffer("membrane_min", membrane_min)

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

	def membrane_reset(self, membrane):
		# Если surrogate_disable, то reset принимает бинарные значения 

		membrane_shift = membrane - self.threshold
		
		# Осоединение от графа вычислений
		reset = self.spike_grad(membrane_shift).clone().detach()

		return reset
	
	def _init_membrane(self):
		membrane = self.membrane_zero.clone()
		self.register_buffer("membrane", membrane, False)

	# Требует доработки, так как необходимо еще и взаимодействовать с буфером, чтобы правильно сбросить потенциал
	def reset_membrane(self):
		# Вероятно обновление вида zeros_like(self.membrane)
		self.membrane = self.membrane_zero.clone()

		return self.membrane
			
	def init_neuron(self):
		return self.reset_membrane()
	
	# Поработать над этим 
	def forward(self, input, membrane = None):
		if membrane != None:
			self.membrane = membrane

		if self.init_hidden and membrane != None:
			raise TypeError("Membrane shouldnt be passed while init_hidden is true")
		# Изменение размера для обучения батчами
		if self.membrane.shape != input.shape:
			self.membrane = torch.ones_like(input) * self.membrane_zero

		self.reset = self.membrane_reset(self.membrane)
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

	# Вероятно использование кумулятивной суммы
	def _membrane_update_function(self, input):
		# Ограничение значений beta в соответствии с моделью. Проводить инплейс операции нельзя, так что нужно придумать альтернативу. 
		update = self.membrane + self.membrane * (-self.beta.clamp(0, 1) + input) * self.dt
		return update


	def _membrane_set_to(self, input):
		# При reset близком к 0 membrane_zero устанавливается на membrane_min
		self.membrane = self.membrane - self.reset * (self.membrane - self.membrane_min) 

		return self._membrane_update_function(input)
		

	def set_dt(self, dt):
		if (isinstance(dt, float) | isinstance(dt, int)):
			if (dt > 0):
				self.dt = dt
			else:
				raise ValueError("dt must be not negative")
		else:
			raise ValueError("dt must be integer of float")
		
	def get_dt(self, dt):
		return self.dt

	# добавить detach_hidden, reset_hidden

	@classmethod
	def init(cls):
		cls.instances = []

	@staticmethod
	def detach(*args):
		for state in args:
			state = torch.zeros_like(state)
			
	@staticmethod
	def _surrogate_bypass(input):
		return (input > 0).float()