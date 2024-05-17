import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MLP, self).__init__()
        def make_layer(in_size, out_size):
            layer = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.LeakyReLU(),
                # nn.Dropout(0.1)
            )
            nn.init.kaiming_normal_(layer[0].weight, nonlinearity='leaky_relu')
            return layer
        self.layers = nn.Sequential(
            make_layer(input_size, hidden_sizes[0]),
            *[make_layer(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], 1),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        return self.layers(x).reshape(-1)
    

class ParticlePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ParticlePositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding.unsqueeze(0).transpose(0, 1))


    def forward(self, x):
        return x + self.encoding[:x.size(0), :]

    
    
class ParticleEventTransformer(nn.Module):
    def __init__(self, feature_size, embed_size, num_heads, hidden_dim, output_dim, num_layers):
        super(ParticleEventTransformer, self).__init__()
        self.particle_embedding = nn.Linear(feature_size, embed_size)
        self.pos_encoder = ParticlePositionalEncoding(embed_size)
        encoder_layers = TransformerEncoderLayer(embed_size, num_heads, hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embed_size * 19, output_dim)
        self.embed_size = embed_size
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.particle_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.particle_embedding(x)  # [batch_size, 19, embed_size]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, embedding_dim]
        x = self.pos_encoder(x * math.sqrt(self.embed_size))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Switch back to [batch_size, seq_len, embedding_dim]
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.output_layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class particleTransformer(nn.Module):
    def __init__(self, particle_feature_size, d_model, nhead, num_encoder_layers, embed_dim, max_seq_length, pos_dropout, layer_widths):
        super().__init__()  
        self.d_model = d_model
        self.embed_src = nn.Linear(particle_feature_size, d_model)
        self.embed_tgt = nn.Linear(particle_feature_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        #self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.NPART = max_seq_length

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.5, inplace = False))
            return layers


        #layer_widths = [200,50,10]
        self.fcblock = nn.Sequential(
                                     *block(d_model*max_seq_length, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )
        #print(self.fcblock)


    def forward(self, src):

        src = src.permute(1,0,2)
        #tgt = tgt.permute(1,0,2)

        #src = self.embed_src(src)
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        #tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        #output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
        #                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.transformer_encoder(src)
        output = output.permute(1,0,2)
        output = output.reshape(-1,self.d_model*self.NPART)
        #print(self.fcblock)
        output = self.fcblock(output)
        return output