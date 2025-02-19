import torch
import torch.nn as nn
# from PyEMD import EEMD
from models.layers import MLP, social_transformer, st_encoder
from pytorch_wavelets import DWT1DForward


class WaveTrans_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 6
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)

		return output_x, state_x.squeeze(0)

class WaveTransformer(nn.Module):
	def __init__(self, t_h: int = 10, hidden_size=[32,64,128,256],):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(WaveTransformer, self).__init__()

		self.social_encoder = social_transformer(t_h)
		self.encoder = WaveTrans_encoder()

		self.dwt_transform = DWT1DForward(J=3)
		self.transformer_encoders = nn.ModuleList([
			nn.TransformerEncoder(
				nn.TransformerEncoderLayer(d_model=hidden_size[i], nhead=2),
				num_layers=2
			) for i in range(4)  # 4 wavelet components
		])

		self.decoder = MLP(256 * 2, 800, hid_feat=(1024, 1024), activation=nn.ReLU())

	def forward(self, x, mask=None):
		'''
		x: batch size, t_p, 6
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256

		output_x, state_x = self.encoder(x)

		# total = torch.cat((state_x, social_embed), dim=-1)

		wavelet_components = self.dwt_transform(state_x.unsqueeze(1))

		component1 = wavelet_components[0]  # (B, L, 32)

		# Second part: (B, L, 128), (B, L, 64), (B, L, 32)
		component2_1 = wavelet_components[1][0]  # (B, L, 128)
		component2_2 = wavelet_components[1][1]  # (B, L, 64)
		component2_3 = wavelet_components[1][2]  # (B, L, 32)

		# We need to prepare a list with all 4 components
		all_components = [component1, component2_3, component2_2, component2_1]

		x0 = self.transformer_encoders[0](all_components[0])
		x1 = torch.cat([x0, all_components[1]], dim=-1)
		x1 = self.transformer_encoders[1](x1)
		x2 = torch.cat([x1, all_components[2]], dim=-1)
		x2 = self.transformer_encoders[2](x2)
		x3 = torch.cat([x2, all_components[3]], dim=-1)
		x3 = self.transformer_encoders[3](x3).mean(dim=1)

		fused_features = torch.cat([x3, social_embed], dim=-1)

		output = self.decoder(fused_features).view(-1, 20, 20, 2)

		return output




