"""An autoregressive Transformer.

This implementation allows for an arbirary ordering of input variables; the
appropriate masking is automatically calculated.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_device

## Notes on masking schemes
#
# Supported values of MASK_SCHEME: 0 (only works for natural ordering), 1 (for
# arbitrary ordering).
#
# Assume we have (x0, x1, x2) and order=natural in all cases below.
#
# Scheme 0 (only works for natural ordering):
#         Input  [SOS, x0, x1]
#         Hidden [p(x0), p(x1|x0), p(x2|x0,x1)]
#      (We use 1 block so hidden == output).
#
#      mask(3) is used with the following result (row=destination index):
#         tensor([[1, 0, 0],
#                 [1, 1, 0],
#                 [1, 1, 1]], dtype=torch.uint8)
#
# Scheme 1 (default):
#   Here, the first attention layer has a different mask than the subsequent
#   layers.  See the detailed docstring of order_respecting_mask().
#
## Notes on unk_embeddings & pos_embeddings
#
#  pos_embeddings index: *position* of input sequence
#    - aganostic to what column index it is (or even if it is SOS)
#
#  unk_embeddings index: *natural_idx* of the column it's supposed to mask
#    - thus, SOS does not have an unk_embeddings
#
#  How they interact: potentially dropout first, then potentially add pos emb.

MASK_SCHEME = 0


# MASK_SCHEME = 1


def mask(n):
    ns = n
    nd = n
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns)
    m = i >= j - ns + nd
    m.requires_grad = False
    return m


def order_respecting_mask(ncols, ordering, input_layer=True):
    """Construct appropriate mask for attention.

    Assuming o=(2,0,1):
     - set inputs = [ SOS=0,          x0,    x1,     x2 ]
     - so outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]

    No one connects to EOS.  SOS connects to everyone.

    Desired mask (row=destination):
        [[1, 0, 0, 1],
         [1, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 0]]

    Mask after the first attention + see self (diagonal)
    Basically == shift above to the left 1 column, then fill diagonal
     - inputs  = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
     - outputs = [ h(x0|x2), h(x1|x0,x2), h(x2), EOS ]
        [[1, 0, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]]
    """
    mask = np.zeros((ncols + 1, ncols + 1))

    if input_layer:
        mask[:, 0] = 1  # First column is SOS -- everyone can see.
        mask[-1, :] = 0  # No one connects to EOS.
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src + 1, ncols):
                # Variable at pos_dst should see pos_src.
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx + 1] = 1
    else:
        for pos_src in range(ncols):
            src_nat_idx = ordering[pos_src]
            for pos_dst in range(pos_src, ncols):
                dst_nat_idx = ordering[pos_dst]
                mask[dst_nat_idx, src_nat_idx] = 1

    mask = torch.as_tensor(mask, dtype=torch.float32)
    mask.requires_grad = False
    return mask


class LayerNorm(nn.Module):
    """Norm to 0-mean 1-std , then do a learned diagonal affine transform."""

    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        s = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(s + self.eps)
        return self.scale * x + self.shift


class Conv1d(nn.Module):
    """Linear with bias add.  Weights ~ N(std), bias ~ 0."""

    def __init__(self, d_in, d_out, w_init_std=0.02):
        super(Conv1d, self).__init__()

        self.w = nn.Parameter(torch.zeros(d_in, d_out))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.w, std=w_init_std)
        nn.init.zeros_(self.b)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):
        *start, d_in = x.size()
        out = torch.matmul(x.view(-1, d_in), self.w) + self.b
        return out.view(start + [self.d_out])


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer.

    Args:
      d_model: last dim of input and output of this module.
      num_heads: number of parallel heads.

    Internally, queries, keys, and values are all produced from the input
    (hence "self"), and all of them are (d_model/num_heads)-dimensional.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_state = d_model // num_heads

        self.qkv_linear = Conv1d(d_model, self.d_state * 3 * num_heads)
        self.linear = Conv1d(num_heads * self.d_state, d_model)

        self.attn_mask = None  # Will be set by caller.

    def _split_heads(self, x):
        # Each input has shape [bs, num cols, d_state * num_heads].
        *start, m = x.size()
        x = x.view(start + [self.num_heads, m // self.num_heads])
        return x.permute(0, 2, 1, 3)

    def _do_attention(self, query, key, value, mask):
        """Accepts Q,K,V each shaped [bs, num heads, num cols, d_state].

        Returns transformed [bs, num_heads, num cols, d_state].
        """
        d_k = query.size()[-1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)
        mask = mask.to(scores.dtype)
        scores = scores * mask - (1 - mask) * 1e10
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, value)
        return out

    def forward(self, x, query_input=None):
        """x: [bs, num cols, d_model].  Output has the same shape."""
        assert x.dim() == 3, x.size()
        bs, ncols, _ = x.size()

        # [bs, num cols, d_state * 3 * num_heads]
        qkv = self.qkv_linear(x)
        # [bs, num heads, num cols, d_state] each
        qs, ks, vs = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        if query_input is not None:
            # TODO: obviously can avoid redundant calc.
            qkv = self.qkv_linear(query_input)
            qs, _, _ = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        # [bs, num heads, num cols, d_state]
        x = self._do_attention(qs, ks, vs, mask=self.attn_mask.to(x.device))

        # [bs, num cols, num heads, d_state]
        x = x.transpose(1, 2)
        # Concat all heads' outputs: [bs, num cols, num heads * d_state]
        x = x.contiguous().view(bs, ncols, -1)
        # Then do a transform: [bs, num cols, d_model].
        x = self.linear(x)
        return x


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    """A Transformer block.

    Args:
      d_model: last dim of input and output of this module.
      d_ff: the hidden dim inside the FF net.
      num_heads: number of parallel heads.
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 num_heads,
                 activation='relu',
                 do_residual=False):
        super(Block, self).__init__()

        self.mlp = nn.Sequential(
            Conv1d(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else GeLU(),
            Conv1d(d_ff, d_model),
        )
        self.norm1 = LayerNorm(features=d_model)
        self.norm2 = LayerNorm(features=d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.do_residual = do_residual

    def set_attn_mask(self, mask):
        self.attn.attn_mask = mask

    def forward(self, x, query_input=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, query_input=query_input)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.do_residual:
            x += residual

        return x


class Transformer(nn.Module):
    """An autoregressive Transformer (decoder only)."""

    def __init__(
            self,
            num_blocks,
            d_model,
            d_ff,
            num_heads,
            nin,
            input_bins,
            use_positional_embs=True,
            activation='gelu',
            fixed_ordering=None,
            seed=None,
            multi_pred_embedding='mlp',
    ):
        """An autoregressive Transformer.

        Namings of the arguments follow from the original paper.

        Args:
          num_blocks: int, number of transformer blocks.
          d_model: int, the hidden dims.
          d_ff: int, each feedforward layer's hidden dims.
          num_heads: int, number of attention heads in each self-attention.
          nin: int, number of input variables.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          use_positional_embs: bool, whether positional encodings are used
            (i.e., whether an input is treated as a sequence or as a set).
          activation: str, the activation function.
          column_masking: if True, turn on column masking during training time,
            which enables the wildcard skipping optimization during inference.
            Recommended to be set for any non-trivial datasets.
          fixed_ordering: variable ordering to use.  Ex: [2, 0, 1] means
            variable 2 is placed in the first position in the autoregressive
            factorization.  If None, either natural ordering (when seed is
            None) or a randomly sampled ordering (otherwise) is used.
          seed: if specified, used for sampling a random ordering.
        """
        super().__init__()

        print('MASK_SCHEME', MASK_SCHEME)
        assert multi_pred_embedding in ['cat', 'mlp', 'rnn', 'rec']
        self.multi_pred_embedding = multi_pred_embedding
        # Common attributes below.
        self.nin = nin
        self.input_bins = input_bins
        encoded_bins = [d_model] * nin
        self.logit_indices = np.cumsum(encoded_bins)
        self.nout = self.logit_indices[-1]
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.embed_size = d_model
        self.emb_dim = d_model
        self.use_positional_embs = use_positional_embs
        assert activation in ['relu', 'gelu']
        self.activation = activation
        self.fixed_ordering = fixed_ordering
        if fixed_ordering is None:
            natural = np.arange(nin)
            if seed is None or seed == 0:
                self.fixed_ordering = natural
            else:
                self.fixed_ordering = np.random.RandomState(seed).permutation(
                    natural)
        print('ordering', self.fixed_ordering)

        # Build.
        self.blocks = nn.Sequential(*[
            Block(d_model,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=(MASK_SCHEME == 0 or i > 0))
            for i in range(num_blocks)
        ])
        # Set masks.
        orig_mask = None
        if MASK_SCHEME == 0:
            orig_mask = mask(nin)
        elif MASK_SCHEME == 1:
            init_attn_mask = order_respecting_mask(nin, self.fixed_ordering)
            attn_mask = order_respecting_mask(nin,
                                              self.fixed_ordering,
                                              input_layer=False)
        else:
            assert False, MASK_SCHEME

        if orig_mask is not None:
            print('using orig mask\n', orig_mask)
            for b in self.blocks:
                b.set_attn_mask(orig_mask)
        else:
            print('init_attn_mask\n', init_attn_mask)
            print('after 1st layer attn_mask\n', attn_mask)
            self.blocks[0].set_attn_mask(init_attn_mask)
            for b in self.blocks[1:]:
                b.set_attn_mask(attn_mask)

        self.input_bins_encoded = list(map(self._get_input_encoded_dist_size, self.input_bins))

        self.norm = LayerNorm(d_model)

        self.embeddings = nn.ModuleList()
        for i in range(nin):
            self.embeddings.append(nn.Embedding(self.input_bins_encoded[i], d_model))
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.02)
        self.embeddings_out = nn.ModuleList()
        for i in range(nin):
            self.embeddings_out.append(nn.Embedding(self.input_bins[i], d_model))
        for e in self.embeddings_out:
            nn.init.normal_(e.weight, std=0.02)
        if use_positional_embs:
            if MASK_SCHEME == 1:
                self.pos_embeddings = nn.Embedding(self.nin + 1, d_model)
            else:
                self.pos_embeddings = nn.Embedding(self.nin, d_model)
            nn.init.normal_(self.pos_embeddings.weight, std=0.01)

        self.unk_embeddings = nn.ParameterList()
        for i, dist_size in enumerate(self.input_bins):
            self.unk_embeddings.append(nn.Parameter(torch.zeros(d_model)))

        # Interface required by ProgressiveSampling.
        self.input_bins_encoded_cumsum = np.cumsum(encoded_bins)
        self.orderings = [self.fixed_ordering]
        self.output_bins_cumsum = np.cumsum(self.input_bins)
        self.multi_pred_embed_nn = None
        self.zero_inp = None
        if self.multi_pred_embedding == 'mlp':
            self.multi_pred_embed_nn = [nn.Sequential(
                nn.Linear(self.d_model, 64),
                nn.Linear(64, self.d_model)
            ) for i in range(len(self.input_bins_encoded))]
        elif self.multi_pred_embedding == 'rec':
            self.multi_pred_embed_nn = [nn.Sequential(
                nn.Linear(self.d_model * 2, 64),
                nn.Linear(64, self.d_model)
            ) for i in range(len(self.input_bins_encoded))]
            self.zero_inp = [torch.zeros(1, self.d_model, requires_grad=False, device=get_device()) for i
                             in range(len(self.input_bins_encoded))]
        elif self.multi_pred_embedding == 'rnn':
            assert False, 'not support since too much cost'

        self.multi_pred_embed_nn = nn.ModuleList(self.multi_pred_embed_nn)

    def _get_input_encoded_dist_size(self, dist_size, apply_multi_preds=False):
        dist_size += 5
        if self.multi_pred_embedding == 'cat' and apply_multi_preds:
            dist_size *= 2
        return dist_size

    def name(self):
        n = 'transformer'
        n += '-blocks' + str(self.num_blocks)
        n += '-model' + str(self.d_model)
        n += '-ff' + str(self.d_ff)
        n += '-heads' + str(self.num_heads)
        if self.use_positional_embs:
            n += '-posEmb'
        n += '-' + self.activation
        # if self.column_masking:
        #     n += '-colmask'
        if MASK_SCHEME == 1:
            n += '-scheme1'
        return n

    def EncodeInput(self, data, preds, natural_col=None, out=None, return_pos_embs=False):
        """Right shift by one token.

        Suppose we want to model x=(x0,x1,x2).
        Set model inputs = [ SOS=0, x0, x1 ]
            (SOS = start of sequence)
        outputs =          [ p(x0); p(x1|x0); p(x2|x0,x1) ].
            (because output i depends on inputs <= i).

        If self.fixed_ordering is supplied and non-natural,
        we set inputs = [ SOS=0, x_o(0), x_o(1) ]
        so    outputs = [ p(x_o(0)), p(x_o(1) | x_o(0)), p(x_o(2) | x_o(0..1)) ]

        This (1) requires when calculating the loss, seq [x_o(0), ..., x_o(2)]
        is passed, (2) assumes we don't change the diagonal attention mask.

        Alternatively (assuming o=(2,0,1)):
          - change diagonal mask to respect ordering o
          - set inputs = [ SOS=0, x_o(0)=x2, x_o(1)=x0 ]
          - so outputs = [ p(x0|x2), p(x1|x0,x2), p(x2) ]
          - doesn't require feeding targets under order o
        """

        def Embed(data, preds, natural_col=None, out=None, batch_mask=None):
            if natural_col is not None:
                assert not return_pos_embs
                # assert preds is not None
                assert out is not None
                # if preds is None:
                #     preds = torch.zeros((1, 5), device=out.device)
                #     preds[:, 0] = 1
                return self.EncodeInputInference(data, preds, natural_col, out)

            if data.dtype != torch.long:
                data = data.long()
            bs = data.size()[0]

            inp_seq_len = self.nin

            if batch_mask is None:
                dropout_vec = torch.dropout(torch.ones(bs,
                                                       inp_seq_len,
                                                       1,
                                                       device=data.device),
                                            p=np.random.randint(0, self.nin) /
                                              self.nin,
                                            train=self.training)
                # During training, non-dropped 1's are scaled by 1/(1-p), so we
                # clamp back to 1.  Shaped [bs, num cols, 1].
                batch_mask = torch.clamp(dropout_vec, 0, 1)
            if MASK_SCHEME == 0:
                # SOS = start of sequence symbol, just zeros.
                y_embed = [torch.zeros(bs, self.embed_size, device=data.device)]
                for nat_idx in range(self.nin - 1):
                    none_pred_mask = (data[:, nat_idx] == -1).view(-1, )
                    pred = torch.clone(preds[:, nat_idx * 5:(nat_idx + 1) * 5])
                    pred[batch_mask[:, nat_idx, :].flatten().long()] = 0
                    tmp = torch.zeros(data.shape[0], self.input_bins[nat_idx], device=data.device)
                    tmp[~none_pred_mask] = torch.scatter(tmp[~none_pred_mask], 1, data[~none_pred_mask, nat_idx].view(-1, 1), 1)
                    emb = torch.matmul(torch.cat([tmp, pred], dim=-1), self.embeddings[nat_idx].weight)
                    emb[none_pred_mask] = self.unk_embeddings[nat_idx]
                    y_embed.append(emb)
            elif MASK_SCHEME == 1:
                y_embed = [torch.zeros(bs, self.embed_size, device=data.device)]
                for nat_idx in range(self.nin):
                    none_pred_mask = (data[:, nat_idx] == -1).view(-1, )
                    pred = torch.clone(preds[:, nat_idx * 5:(nat_idx + 1) * 5])
                    pred[batch_mask[:, nat_idx, :].flatten().long()] = 0
                    tmp = torch.zeros(data.shape[0], self.input_bins[nat_idx], device=data.device)
                    tmp[~none_pred_mask] = torch.scatter(tmp[~none_pred_mask], 1,data[~none_pred_mask, nat_idx].view(-1, 1), 1)
                    emb = torch.matmul(torch.cat([tmp, pred], dim=-1), self.embeddings[nat_idx].weight)
                    emb[none_pred_mask] = self.unk_embeddings[nat_idx]
                    y_embed.append(emb)
            else:
                assert False, MASK_SCHEME

            # [batch size, num cols (+ 1), d_model].  +1 or not depends on scheme.
            inp = torch.stack(y_embed, 1)
            assert inp_seq_len == inp.shape[1]

            # Shaped [1, num cols, d_model].
            dropped_repr = torch.stack(tuple(self.unk_embeddings)).unsqueeze(0)
            if MASK_SCHEME == 0:
                # Namely, [0, unk(0), unk(1)] for ncols=3.  This means:
                #   (1) SOS is never dropped.
                #   (2) indexing into unk_embeddings is based on natural_idx.
                dropped_repr = torch.cat((torch.zeros_like(
                    dropped_repr[:, 0:1, :]), dropped_repr[:, :-1, :]),
                    dim=1)
            else:
                dropped_repr = torch.cat(
                    (torch.zeros_like(dropped_repr[:, 0:1, :]), dropped_repr),
                    dim=1)
            inp = batch_mask * inp + (1. - batch_mask) * dropped_repr

            if self.use_positional_embs:
                # [1, inp_seq_len, d_model]
                # NOTE: indexes into pos embs == positions \in [0, inp_seq_len).
                pos_embs = self.pos_embeddings(
                    torch.arange(inp_seq_len, device=data.device)).unsqueeze(0)
                inp += pos_embs
                if return_pos_embs:
                    return inp, pos_embs, batch_mask
                return inp, None, batch_mask

            assert not return_pos_embs
            return inp, None, batch_mask

        batch_mask = None
        encoded_inps = []
        pos_embs = []
        for i in range(2):
            data_ = data[:, i, :] if data is not None else None
            preds_ = preds[:, i, :] if preds is not None else None
            out_ = out[:, i, :] if out is not None else None
            encoded_input, pos_emb, batch_mask = Embed(data_, preds_, natural_col, out_, batch_mask)
            encoded_inps.append(encoded_input.unsqueeze(1))
            if pos_emb is not None:
                pos_embs.append(pos_emb.unsqueeze(1))
            batch_mask = batch_mask
        encoded_input = torch.cat(encoded_inps, dim=1)
        if len(pos_embs) > 0:
            pos_embs = torch.cat(pos_embs, dim=1)
        else:
            pos_embs = None

        return encoded_input, pos_embs

    def EncodeInputInference(self, data, preds, natural_col, out):
        """Special inference path.

        Args:
          data: [batch size, 1].  Just the data for column 'natural_col'.
          natural_col (int): [0, num cols).
          out: shaped [batch size, d_model].  To hold the encoded data.
        """
        if natural_col < 0:
            # Potentially handling SOS.
            if self.use_positional_embs:
                # Let's also add E_pos=0 to SOS (if enabled).
                out.copy_(
                    self.pos_embeddings(torch.as_tensor(
                        0,
                        device=data.device)).unsqueeze(0).expand(data.size()[0], -1))
            return out, None, None

        if data is None:
            # [bs, d_model]
            embs = self.unk_embeddings[natural_col].unsqueeze(0).expand(
                out.shape[0], -1)
        else:
            none_pred_mask = (data == -1).view(-1, )
            tmp = torch.zeros(data.shape[0], self.input_bins[natural_col], device=data.device)
            tmp[~none_pred_mask] = torch.scatter(tmp[~none_pred_mask], 1, data[~none_pred_mask].long(), 1)
            # [bs, d_model]
            embs = torch.matmul(torch.cat([tmp, preds], dim=-1), self.embeddings[natural_col].weight)
            embs[none_pred_mask] = self.unk_embeddings[natural_col]
            # embs = self.embeddings[natural_col](x).squeeze(1)

        if self.use_positional_embs:
            # NOTE: this is tricky.  Under MASK_SCHEME=0 or 1, E_pos=0 is added
            # to SOS, E_pos=1 is added to x0, etc.  So we need to take this into
            # account.
            pos = self.pos_embeddings(
                torch.as_tensor(natural_col + 1,
                                device=out.device)).unsqueeze(0)
            embs = embs + pos

        out.copy_(embs)
        return out, None, None

    def merge_multi_preds(self, x):
        # x is a list of length self.nin, each x[i].shape == (batch, 2, encode_size)
        if isinstance(x, torch.Tensor):
            x = torch.split(x, self.input_bins_encoded, dim=-1)
        out = []
        if self.multi_pred_embedding == 'cat':
            return torch.cat(list(map(lambda t: t.flatten(start_dim=1), x)), dim=-1)

        def run_rec(x1, x2):
            # x1 ,x2 shape is (bs, encoded)
            if x1.shape[0] != x2.shape[0]:
                x1 = torch.tile(x1, dims=(x2.shape[0], 1))
            x = torch.cat([x1, x2], dim=-1)
            return self.multi_pred_embed_nn[i](x)

        for i in range(self.nin):
            if self.multi_pred_embedding == 'mlp':
                out.append(torch.sum(self.multi_pred_embed_nn[i](x[i]), dim=1))
            elif self.multi_pred_embedding == 'rnn':
                tmp = self.multi_pred_embed_nn[i][:-1](x[i])[0]
                out.append(torch.sum(self.multi_pred_embed_nn[i][-1](tmp), dim=1))
            elif self.multi_pred_embedding == 'rec':
                x1 = self.zero_inp[i]
                for j in range(x[i].shape[1]):
                    x1 = run_rec(x1, x[i][:, j, :])
                out.append(x1)
        return torch.stack(out, dim=1)

    def forward(self, x):
        """Outputs logits for (x0, x1|x0, x2|x0,x1, ...)."""
        # [bs, ncols] -> [bs, ncols, d_model].  Right-shifted.
        if MASK_SCHEME == 1:
            assert self.use_positional_embs, 'should enable positional embs'
            x, pos_embs = self.EncodeInput(*x, return_pos_embs=True)
            x = list(map(lambda x: x.squeeze(), torch.split(x, 1, dim=2)))
            x = self.merge_multi_preds(x)
            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x, _ = self.EncodeInput(*x)
            x = list(map(lambda x: x.squeeze(), torch.split(x, 1, dim=2)))
            x = self.merge_multi_preds(x)
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def forward_with_encoded_input(self, x, decode=True, ordering=None):
        x = torch.split(x, self.d_model, dim=-1)
        x = self.merge_multi_preds(x)
        # [batch size, num cols * d_model] -> [bs, num cols, d_model]
        x = x.view(x.shape[0], -1, self.d_model)

        if MASK_SCHEME == 1:
            inp_seq_len = x.shape[1]

            assert self.use_positional_embs, 'Need pos_embs for 1st layer query vecs'
            pos_embs = self.pos_embeddings(
                torch.arange(inp_seq_len, device=x.device)).unsqueeze(0)

            x = self.blocks[0](x, query_input=pos_embs)
            for b in self.blocks[1:]:
                x = b(x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        if decode:
            out = []
            assert ordering is not None
            for i in range(self.nin):
                natural_idx = ordering[i]
                out.append(self.logits_for_col(natural_idx, x))
            x = torch.cat(out, dim=-1)
        return x

    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, ncols+1, d_model].
          data: [batch size, ncols].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            ce = F.cross_entropy(logits_i, data[:, i], reduction='none')
            nll += ce
        return nll

    def logits_for_col(self, idx, logits, is_training=False):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, ncols+1, d_model].

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        embed = self.embeddings_out[idx]
        return torch.matmul(logits[:, idx, :], embed.weight.t())


if __name__ == '__main__':
    num_cols = 3
    vocab = 1
    bs = 1
    num_cols = 11
    vocab = 5
    bs = 3
    orderings = [
        np.arange(num_cols),
        np.arange(num_cols)[::-1],
        np.random.permutation(np.arange(num_cols)),
    ]
    for ordering in orderings:
        print('Testing ordering', ordering)
        model = Transformer(num_blocks=2,
                            d_model=16,
                            d_ff=64,
                            num_heads=4,
                            nin=num_cols,
                            input_bins=[
                                           vocab,
                                       ] * num_cols,
                            use_positional_embs=True,
                            activation='gelu',
                            fixed_ordering=ordering)
        print('attn_mask for blk 0', model.blocks[0].attn.attn_mask)

        for i in range(num_cols):
            nat_idx = ordering[i]
            print('\nchecking output column {} nat_idx {}...'.format(
                i, nat_idx))
            inp = torch.randint(vocab, (bs, num_cols))
            # [bs, num cols, d_model], the logits
            out = model(inp)

            out[:, nat_idx, :].contiguous().view(-1, )[0].backward()
            ok = True
            for n, p in model.named_parameters():
                if 'embed' in n:
                    if p.grad is None:
                        print(n, p.grad)
                        continue
                    dep = (p.grad.reshape(-1) != 0).numpy().any()
                    for j in range(i + 1, len(ordering)):
                        nat_idx_j = ordering[j]
                        # i.e., p corresponds to nat_idx j
                        if n == 'embeddings.{}.weight'.format(nat_idx_j):
                            ok &= (not dep)
            assert ok

        print('[Transformer] Passes autoregressive-ness check!')
