import torch
from sirarm_template.utils.torch import is_parallel_model


class ModelEMA:
	"""
	Exponential Moving Average (EMA) of model parameters.
	"""
	
	def __init__(self, model, decay: float = 0.999):
		self.model = model
		self.decay = decay
		self.shadow = {}
		self.backup = {}
		
		self._init_shadow()
	
	def get_model(self):
		if is_parallel_model(self.model):
			_model = self.model.module
		else:
			_model = self.model
		return _model
	
	def _init_shadow(self):
		_model = self.get_model()
		for name, param in _model.named_parameters():
			if param.requires_grad:
				if self.shadow is None:
					self.shadow = {}
				self.shadow[name] = param.data.clone()
	
	@torch.no_grad()
	def update(self):
		_model = self.get_model()
		for name, param in _model.named_parameters():
			if name in self.shadow:
				self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
	
	@torch.no_grad()
	def apply_shadow(self):
		_model = self.get_model()
		for name, param in _model.named_parameters():
			if name in self.shadow:
				self.backup[name] = param.data.clone()
				param.data.copy_(self.shadow[name])
	
	@torch.no_grad()
	def restore(self):
		_model = self.get_model()
		for name, param in _model.named_parameters():
			if name in self.backup:
				param.data.copy_(self.backup[name])
		self.backup.clear()
	
	def state_dict(self):
		return {k: v.clone() for k, v in self.shadow.items()}
	
	def load_state_dict(self, state_dict):
		self.shadow = {k: v.clone() for k, v in state_dict.items()}
