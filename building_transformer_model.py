import torch
import torch.nn as nn
import math

# Setting up the device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer model parameters
d_model = 512  # The number of expected features in the encoder/decoder inputs
nhead = 8  # The number of heads in the multiheadattention models
num_encoder_layers = 3  # The number of sub-encoder-layers in the encoder
num_decoder_layers = 3  # The number of sub-decoder-layers in the decoder
dim_feedforward = 2048  # The dimension of the feedforward network model
dropout = 0.1  # The dropout value

# Sample tokenizers (these should be replaced with the actual tokenizers for your languages)
src_language = "en"
tgt_language = "fr"

# Replace these with the actual vocabulary sizes for your source and target languages
src_vocab_size = 10000
tgt_vocab_size = 10000


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Instantiate the model
transformer_model = TransformerModel(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
).to(device)

# Example input batch
src = torch.rand((10, 32)).long().to(device)  # (source sequence length, batch size)
tgt = torch.rand((20, 32)).long().to(device)  # (target sequence length, batch size)

# Masks and padding
src_mask = transformer_model.transformer.generate_square_subsequent_mask(
    src.size(0)
).to(device)
tgt_mask = transformer_model.transformer.generate_square_subsequent_mask(
    tgt.size(0)
).to(device)
src_padding_mask = (src == 0).transpose(0, 1).to(device)
tgt_padding_mask = (tgt == 0).transpose(0, 1).to(device)
memory_key_padding_mask = src_padding_mask

# Forward pass
outputs = transformer_model(
    src,
    tgt,
    src_mask,
    tgt_mask,
    src_padding_mask,
    tgt_padding_mask,
    memory_key_padding_mask,
)

print(outputs.shape)  # (target sequence length, batch size, target vocabulary size)
