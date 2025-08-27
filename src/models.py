import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *

# for iTransformer
from layers.Transformer_EncDec import Encoder, EncoderLayer, Encoder_AnomalyTransformer, EncoderLayer_AnomalyTransformer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, AnomalyAttention
from layers.Embed import DataEmbedding_inverted, DataEmbedding


## LSTM_AE Model (based on IEEE SOCA 2019, Hsieh et al.)
class LSTM_AE(nn.Module):
	def __init__(self, feats, window_size=10):
		super(LSTM_AE, self).__init__()
		self.name = 'LSTM_AE'
		self.lr = 0.002
		self.batch = 100
		self.n_feats = feats
		self.n_hidden = 64
		self.window_size = window_size
		self.lstm = nn.LSTM(input_size=self.window_size, hidden_size=self.n_hidden, num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=1, batch_first=True)
		self.fcn = nn.Linear(self.n_hidden, self.n_feats)

	def forward(self, x):	
		x = x.permute(0, 2, 1)
		out1, (h1, c1) = self.lstm(x)
		latent = h1.repeat(self.window_size, 1, 1)
		latent = h1.repeat(self.window_size, 1, 1)
		latent = latent.swapdims(0, 1)
		out2, (h2, c2) = self.lstm2(latent)
		out2 = self.fcn(out2)
		outputs = out2
		return outputs

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats, window_size=None):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.batch = 1
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(self.n_feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats,  window_size=5):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.batch = 1
		self.window_size = window_size # USAD w_size = 5
		self.n = self.n_feats * self.window_size
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

# TranAD + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats, window_size):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 24 
		self.n_feats = feats
		self.window_size = window_size
		self.n = self.n_feats * self.window_size
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.window_size)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  #, enable_nested_tensor=False)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# iTransformer (ICLR 2024)
class iTransformer(nn.Module):
	def __init__(self, feats, window_size, d_model=2, forecasting=False):
		super(iTransformer, self).__init__()
		self.name = 'iTransformer'
		self.lr = 0.0001
		self.batch = 32  
		self.n_feats = feats
		self.window_size = window_size
		self.n = self.n_feats * self.window_size
		self.seq_len = self.window_size
		self.label_len = self.window_size
		self.forecasting = forecasting 		# whether model output should be reconstruction or forecasting of input
		if self.forecasting:
			self.pred_len = 1  				# for forecasting-based AD, only want to predict 1 step ahead
		else:
			self.pred_len = self.window_size
		self.output_attention = False
		self.use_norm = True
		self.d_model = d_model 
		self.embed = 'TimeF'
		self.freq = 's'
		self.dropout = 0.1
		self.n_heads = feats  
		self.e_layers = 2
		self.d_ff = 256 
		self.factor = 1  # attention factor
		self.activation = 'gelu'
        # Embedding
		self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        # Encoder-only architecture
		self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), 
									  self.d_model, self.n_heads, d_keys=self.d_model, d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
		self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

	def forecast(self, x_enc, x_mark_enc=None):
		if self.use_norm:
			# Normalization from Non-stationary Transformer
			means = x_enc.mean(1, keepdim=True).detach()
			x_enc = x_enc - means
			stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
			x_enc /= stdev

		_, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
		enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
		enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N  
		dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

		if self.use_norm:
			# De-Normalization from Non-stationary Transformer
			dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
			dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

		return dec_out
	
	def forward(self, src, src_mark_enc=None):
		dec_out = self.forecast(src, src_mark_enc) 
		out = dec_out
		return out

# vanilla transformer for comparison, same structure as iTransformer except for the embedding
class Transformer(nn.Module):  
	def __init__(self, feats, window_size, d_model=2):
		super(Transformer, self).__init__()
		self.name = 'Transformer'
		self.lr = 0.0001
		self.batch = 12  
		self.n_feats = feats
		self.window_size = window_size
		self.seq_len = self.window_size
		self.label_len = self.window_size
		self.pred_len = self.window_size  
		self.output_attention = False
		self.use_norm = True
		self.d_model = d_model 
		self.embed = 'TimeF'
		self.freq = 's'
		self.dropout = 0.1
		self.n_heads = feats  
		self.e_layers = 2
		self.d_ff = 256 
		self.factor = 1  # attention factor
		self.activation = 'gelu'
        # Embedding, NOT inverted
		self.enc_embedding = DataEmbedding(self.n_feats, self.d_model, self.embed, self.freq, self.dropout)
        # Encoder-only architecture
		self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), 
									  self.d_model, self.n_heads, d_keys=self.d_model, d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
		self.projector = nn.Linear(self.d_model, self.n_feats, bias=True)

	def forecast(self, x_enc, x_mark_enc=None):
		if self.use_norm:
			# Normalization from Non-stationary Transformer
			means = x_enc.mean(1, keepdim=True).detach()
			x_enc = x_enc - means
			stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
			x_enc /= stdev

		_, L, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B L E                in the vanilla Transformer)
		enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B L E -> B L E                in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
		enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B L E -> B L N
		dec_out = self.projector(enc_out)

		if self.use_norm:
			# De-Normalization from Non-stationary Transformer
			dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
			dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

		return dec_out
	
	def forward(self, src, src_mark_enc=None):
		dec_out = self.forecast(src, src_mark_enc)  
		out = dec_out
		return out
	


# Anomaly Transformer (ICLR 2022)
class AnomalyTransformer(nn.Module):
	def __init__(self, feats, window_size, d_model=2):
		super(AnomalyTransformer, self).__init__()
		self.name = 'AnomalyTransformer'
		self.lr = 0.0001
		self.batch = 32
		self.feats = feats
		self.window_size = window_size
		self.output_attention = True
		self.d_model = d_model 
		self.n_heads = 8
		self.e_layers = 3
		self.d_ff = 64
		self.dropout = 0.0
		self.activation = 'gelu'

		# Encoding
		self.embedding = DataEmbedding(self.feats, self.d_model, self.dropout)

		# Encoder
		self.encoder = Encoder_AnomalyTransformer(
			[
				EncoderLayer_AnomalyTransformer(
					AttentionLayer(
						AnomalyAttention(self.window_size, False, attention_dropout=self.dropout, 
										 output_attention=self.output_attention),
						self.d_model, self.n_heads),
					self.d_model,
					self.d_ff,
					dropout=self.dropout,
					activation=self.activation
				) for l in range(self.e_layers)
			],
			norm_layer=torch.nn.LayerNorm(self.d_model)
		)

		self.projection = nn.Linear(self.d_model, self.feats, bias=True)  # should output be in shape of  window size or feats?

	def forward(self, x, x_mark_enc=None):
		enc_out = self.embedding(x, x_mark_enc)
		enc_out, series, prior, sigmas = self.encoder(enc_out)
		enc_out = self.projection(enc_out)

		if self.output_attention:
			return enc_out, series, prior, sigmas
		else:
			return enc_out  # [B, L, D]
