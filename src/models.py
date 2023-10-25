from enum import Enum, auto
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model

from encoder import Encoder


class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
        input : [ B x n_models x input_dim ]
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum("eoi,bei->beo", self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output


class GRUPosition(Enum):
    BEFORE = auto()
    AFTER = auto()
    NEITHER = auto()


class GPT(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        gpt2_args: dict,
        gru_position: str,
        n_embd: int,
        n_tokens: int,
        step_dim: int,
        steps_per_context: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.gru_position = GRUPosition[gru_position.upper()]
        context_size = steps_per_context * step_dim - 1

        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(n_tokens * step_dim + 1, n_embd)

        # transformer
        # Use Huggingface's GPT2:
        config = GPT2Config(
            vocab_size=n_tokens + 1,
            n_positions=context_size,
            n_ctx=context_size,
            n_embd=n_embd,
            **gpt2_args
        )
        self.gru = (
            None
            if self.gru_position == GRUPosition.NEITHER
            else nn.GRU(n_embd, n_embd, batch_first=True)
        )
        self.gpt2_model = GPT2Model(config)
        # decoder head
        self.head = EinLinear(step_dim, n_embd, n_tokens + 1, bias=False)

        self.vocab_size = n_tokens
        self.stop_token = n_tokens * step_dim
        self._context_size = context_size
        self.transition_dim = step_dim

        self.embedding_dim = n_embd
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def offset_tokens(self, idx):
        _, t = idx.shape
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(idx.device)
        offset_idx = idx + offsets[:t]
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.transition_dim))
        inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t]
        for i in range(self.transition_dim):
            x_ = x[:, inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:, i].view(b, n_states, embedding_dim)[:, :t_]
            print(i, x_.shape, x_pad_.shape)
            assert (x_ == x_pad_).all()

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        """
        sequence : [ B x (T+1) ]
        """
        inputs = sequence[:, :-1].contiguous()
        targets = sequence[:, 1:].contiguous()

        inputs = self.encoder.encode(inputs)

        b, t = inputs.size()
        assert t <= self._context_size, "Cannot forward, model block size is exhausted."

        offset_idx = self.offset_tokens(inputs)
        # [ B x T x embedding_dim ]
        # forward the GPT model
        x = self.tok_emb(offset_idx)  # each index maps to a (learnable) vector
        if self.gru_position == GRUPosition.BEFORE:
            x, _ = self.gru(x)

        # [ B x T x embedding_dim ]
        x = self.gpt2_model(inputs_embeds=x).last_hidden_state
        # [ B x T x embedding_dim ]
        if self.gru_position == GRUPosition.AFTER:
            x, _ = self.gru(x)

        # [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad, n_pad = self.pad_to_full_observation(x)
        # [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head(x_pad)
        # [ B x T' x (vocab_size + 1) ]
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        # [ B x T x (vocab_size + 1) ]
        logits = logits[:, :t]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            self.encoder.encode(targets).view(-1),
            reduction="none",
        )

        # if we are given some desired targets also calculate the loss
        if weights is None:
            loss = loss.mean()
        else:
            loss = loss * weights[:, 1:].reshape(-1)
            if mask is not None:
                mask = mask[:, 1:].contiguous()
                loss = (loss * mask.view(-1)).mean()

        return logits, loss

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        prediction = torch.multinomial(probs, num_samples=1)
        return self.encoder.decode(prediction)
