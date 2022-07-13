import numpy as np
import math
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data
import collections
import copy

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1E9):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    """
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(logits, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).sum(-1, keepdim=True)

        #shift and sort
        shifted_scores = torch.abs((-ent) - normalized)
        _, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
      
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = sorted_indices_to_remove.long()

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0

        sorted_indices = torch.tensor(sorted_indices.cpu().data.numpy())
        shp = logits.shape
        logits_red = logits.reshape((-1, shp[-1]))
        sorted_indices_red = sorted_indices.reshape((-1, shp[-1]))
        sorted_indices_to_remove_red = sorted_indices_to_remove.reshape((-1, shp[-1]))
        for i in range(shp[0]):
            logits_red[i][sorted_indices_red[i]] = logits_red[i][sorted_indices_red[i]] * (1. - sorted_indices_to_remove_red[i]) + sorted_indices_to_remove_red[i] * filter_value
        logits = logits_red.reshape(shp)
    return logits


def typical_top_k_filtering(logits, top_k=0, top_p=0.0, temperature=1.0, min_tokens_to_keep=1, filter_value=-1E12):
    """ Filter a distribution of logits using typicality, with optional top-k and/or nucleus (top-p) filtering
        Meister et. al. https://arxiv.org/abs/2202.00666
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep top k tokens with highest prob (top-k filtering).
            top_p >0.0: keep the top p tokens which compose cumulative probability mass top_p (nucleus filtering).
            min_tokens_to_keep >=1: always keep at least this many tokens through the top_p / nucleus sampling
    """
    # https://arxiv.org/abs/2202.00666
    # based on hugging face impl but added top k
    # https://github.com/cimeister/typical-sampling/commit/0f24c9409dc078ed23982197e8af1439093eedd3#diff-cde731a000ec723e7224c8aed4ffdedc9751f5599fe0a859c5c65d0c5d94891dR249
    # changed some of the scatter logic to looping + stacking due to spooky threaded cuda errors, need to CUDA_NONBLOCKING=1 to fix

    # typical decoding
    scores = logits
    mass = top_p if top_p > 0.0 else 1.0
    # calculate entropy
    log_p = torch.nn.functional.log_softmax(scores, dim=-1)
    p = torch.exp(log_p)
    ent = -(p * log_p).sum(-1, keepdim=True)
    # shift and sort
    # abs(I() - H())
    # I() is -log(p()) from eq 5
    # so overall we see -log(p()) - ent
    # orig code was ((-ent) - log_p) 
    shifted_scores = torch.abs(-log_p - ent)

    # possible to calculate the scores over k steps? ala classifier free guidance / CLIP guides?

    # most typical (0) to least typical (high abs value)
    _, sorted_indices = torch.sort(shifted_scores, descending=False, stable=True)
    top_k = min(top_k, scores.size(-1) - 1)  # safety check that top k is not too large
    # this semi-butchers some of the core arguments of the paper, but top k can be good
    # think of this as doing typical decoding / reordering based on the top k by prob
    # top k by typicality seems to be kinda weird for music?
    #if top_k > 0:
    #    topkval = torch.topk(scores, top_k)[0][..., -1, None]
    #    indices_to_remove = scores < topkval
    #    scores[indices_to_remove] = filter_value
    if top_k > 0:
        topkval = torch.topk(torch.max(shifted_scores) - shifted_scores, top_k)[0][..., -1, None]
        indices_to_remove = (torch.max(shifted_scores) - shifted_scores) < topkval
        scores[indices_to_remove] = filter_value
   
    sorted_scores = scores.gather(-1, sorted_indices)
    cumulative_probs = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens once cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > mass
    sorted_indices_to_remove = sorted_indices_to_remove.long()
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_indices = torch.tensor(sorted_indices.cpu().data.numpy())
    shp = scores.shape
    # not great cuda errors on gather calls here, rewrote to a "slow" version
    scores_red = scores.reshape((-1, shp[-1]))
    sorted_indices_red = sorted_indices.reshape((-1, shp[-1]))
    sorted_indices_to_remove_red = sorted_indices_to_remove.reshape((-1, shp[-1]))
    for i in range(shp[0]):
        scores_red[i][sorted_indices_red[i]] = scores_red[i][sorted_indices_red[i]] * (1. - sorted_indices_to_remove_red[i]) + sorted_indices_to_remove_red[i] * filter_value
    scores = scores_red.reshape(shp)
    return scores

device = torch.device(device_str)
_device_default = device #"cuda"
def get_device_default():
    return _device_default

def set_device_default(dd):
    _device_default = dd

_dtype_default = torch.float32
def get_dtype_default():
    return _dtype_default

def set_dtype_default(dt):
    _dtype_default = dt


class RampOpt(object):
    """
    Similar to NoamOpt but with specified "max" learning rate rather than derived from model size
    Factor specifies whether ramp linearly or to the X power
    Warmup describest the number of steps to ramp up learning rate
    Decay to 0 by default, using cosine decay
        terminates at decay_to_zero_at_steps
    RampOpt(target_learning_rate, ramp_power, steps, opt)
    return RampOpt(.0001, 1, 4000, 4000 * 100,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    can set decay_to_zero_at_steps to -1 to disable decay
    """
    def __init__(self, target_learning_rate, factor, warmup, decay_to_zero_at_steps, optimizer, min_decay_learning_rate=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.target_learning_rate = target_learning_rate
        self.decay_to_zero_at_steps = decay_to_zero_at_steps
        self.min_decay_learning_rate = min_decay_learning_rate
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step <= self.warmup:
            return self.target_learning_rate * ((step / float(self.warmup)) ** self.factor)

        if self.decay_to_zero_at_steps == -1:
            return self.target_learning_rate

        new_rate = self.target_learning_rate * np.cos((float(step - self.warmup) / (self.decay_to_zero_at_steps - self.warmup)) * (np.pi / 2.))

        if self.min_decay_learning_rate is not None:
            if new_rate < self.min_decay_learning_rate:
                new_rate = self.min_decay_learning_rate

        if step > self.decay_to_zero_at_steps:
            if self.min_decay_learning_rate is None:
                print("WARNING: RampOpt optimizer has decayed to LR 0! Current step {}, so no more learning happening!".format(step))
                new_rate = 0.
        # warmup is 0 on cos curve
        # infinity is pi/2?
        return new_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


def clipping_grad_norm_(parameters, rescale, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print(p.grad.data.sum())
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    grad_norm = torch.sqrt(sum([torch.sqrt(torch.pow(p.grad.data, 2).sum()) for p in _params]))
    scaling_num = rescale
    scaling_den = max([1.0 * rescale, grad_norm])
    scaling = scaling_num / scaling_den
    for p in _params:
        p.grad.data.mul_(scaling)

def clipping_grad_value_(parameters, clip_value, named_parameters=False, named_check=False):
    # is a generator... get a static reference so the second iteration isn't empty
    if named_check:
        for n, p in parameters:
            print("Checking {} grad.data".format(n))
            assert p.grad.data is not None
            print(p.grad.data.sum())
            print("{}, OK".format(n))
        raise ValueError("named_check complete!")
    if not named_parameters:
        _params = [p for p in parameters]
    else:
        _params = [p[1] for p in parameters]

    clip_value = float(clip_value)
    for p in _params:
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


def get_sequence_lengths(inputs):
    # can be 0 when it is max length (no 0 present) or pos 0 is 0
    lengths = torch.argmax((inputs == 0).int(), axis=0)
    # this handles edge case when no 0 but everything is full
    new_lengths = torch.where(torch.logical_and(lengths == 0, inputs[0] != 0), inputs.shape[0], lengths)
    return new_lengths

def make_block_causal_masks(inputs,
                            latent_index_dim, 
                            batch_size,
                            latents_per_position=1):
    input_index_dim = inputs.shape[0]
    def get_steps(inputs):
        lengths = get_sequence_lengths(inputs)
        num_unique_positions = latent_index_dim
        assert input_index_dim >= num_unique_positions
        last_steps = lengths[:, None] * torch.ones(num_unique_positions).to(inputs.device)[None]
        offsets = torch.arange(-num_unique_positions, 0, step=1).to(inputs.device)
        last_steps += offsets
        last_steps = torch.maximum(last_steps, -1 + 0. * last_steps)
        return last_steps
    latent_last_steps = get_steps(inputs)
    input_ids = torch.arange(input_index_dim).to(inputs.device)[:, None]
    # needs to be multi-dimensional mask here
    # T B latent
    encoder_mask_raw = (input_ids[..., None] <= latent_last_steps[None]).int()


    # latent last steps encodes the last step in the encoder seq that is entering the latent at that position
    # latent last steps is T B latents
    # tensor([[ 9., 10., 11.],
    #    [ 8.,  9., 10.],
    #    [ 7.,  8.,  9.],
    #    [ 6.,  7.,  8.]])
    # this would mean that at batch el 0, the 0th latent holds info about things up to timestep 9 of the input
    # then first latent up to 10, second latent up to 11 (for 3 latents)
    # we see a difference for the other batch elements (batch size 4) because they are different length, different padding

    #print(latent_last_steps)
    #print(input_ids)
    # hack
    #for _i in range(batch_size):
    #    encoder_mask_raw[:int(latent_last_steps[_i, 0]), :, :] = 1
    # current mask setup is 1.0 allow, 0.0 masked / block
    #print(encoder_mask_raw[9, 0])    

    valid_inputs = (inputs > 0).int()
    encoder_mask_final = encoder_mask_raw * valid_inputs[..., None]

    # from annotated transformer    
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=0).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 1
    
    processor_mask = subsequent_mask(latent_index_dim).to(inputs.device).int().permute(1, 0, 2)
    # broadcast to batch size, will be latent B latent
    # need to be careful here about rows vs columns

    processor_mask = processor_mask + (0. * encoder_mask_final[:1, :, :1])
    #print(processor_mask.shape)
    #print(processor_mask[:, 0, :])

    # loss mask will remove the grads through "invalid" latents
    # processor mask is latent_dim, batch, latent_dim
    # TODO: dropout here in mask, or handle later?
    # encoder mask is cross-attend size, batch, latent size
    # processor mask is latent size, batch, latent size
    # interpret processor mask as which latents each step can see e.g.
    # p_m[:, 0, 0] shows which latents output batch el 0, step 0 can look at
    # should be something like [1, 0, 0] for latent size 3
    # same interp for encoder but because xattend has different dim usually there is less confusion
    return (encoder_mask_final, processor_mask), latent_last_steps

def make_rotation_matrices(
    x,
    max_wavelength,
    positions,
):
  """Builds the cosine and sine matrices used to compute rotary embeddings.
  Args:
    x: The array the rotary embeddings will be applied to [T, B, n_heads, head_dim].
    max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
      This specifies the maximum sequence length for identifying unique
      positions.
    positions: A [T, B] tensor of positions.
  Returns:
    cos_matrix: [B, 1, T, head_dim] cosine component of the embedding rotation.
    sin_matrix: [B, 1, T, head_dim] sine component of the embedding rotation.
  """
  seq_len, batch_size, _, head_dim = x.shape

  # head_dim is assumed to be constructed/padded so it's even
  assert head_dim % 2 == 0

  # Generated log-spaced wavelengths between 1 and the max_wavelength.
  num_bands = head_dim // 2
  freq = max_wavelength**((2./head_dim)*torch.linspace(0, num_bands, num_bands, device=x.device))
  inv_freq = 1./freq
  inv_freq = torch.repeat_interleave(inv_freq, 2, axis=0)  # 2x for sin / cos
  positions = positions.permute((1, 0))
  radians = torch.einsum('bi,j -> bij', positions, inv_freq)  # [T, head_dim]
  radians = torch.reshape(radians, (batch_size, 1, seq_len, head_dim))
  #radians = radians.permute((2, 1, 0, 3))
  return torch.cos(radians), torch.sin(radians)

def splice_array(x):
  """Reorders the embedding dimension of an array, to make rotation easier."""
  # head_dim is assumed to be constructed/padded so it's even
  assert x.shape[-1] % 2 == 0

  even_dims = x[..., ::2]
  odd_dims = x[..., 1::2]
  return torch.stack((-odd_dims, even_dims), axis=-1).reshape(x.shape)

def apply_rotary_encoding(x,
                          max_wavelength,
                          positions):
  """Applies the rotary embedding matrix to an input array.
  Computes R*x, the multiplication between the rotation matrix R, and input x.
  Args:
    x: Array of shape [T, B, num_heads, head_dim]
    max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
      This specifies the maximum sequence length for identifying unique
      positions.
    positions: A [T, B] tensor of positions.
  Returns:
    Array of rotary encoded input, of shape [T, B, num_heads, head_dim].
  """
  # {cos, sin}_matrix are [B, 1, T, head_dim]
  cos_matrix, sin_matrix = make_rotation_matrices(
      x, max_wavelength, positions)
  # [T, B, num_heads, head_dim] -> [B, num_heads, T, head_dim]
  x = x.permute(1, 2, 0, 3)
  # Apply the rotation.
  rotary_embeddings = x * cos_matrix + splice_array(x) * sin_matrix
  # [B, num_heads, T, head_dim] -> [T, B, num_heads, head_dim]
  return rotary_embeddings.permute(2, 0, 1, 3)


def apply_rotary_encoding_to_subset(
    x,
    fraction_to_rotate,
    fraction_heads_to_rotate,
    max_wavelength,
    positions):
  """Applies a rotary positional encoding to a subset of dimensions."""
  if fraction_to_rotate > 1.0 or fraction_to_rotate <= 0.0:
    raise ValueError(
        f'fraction_to_rotate must be in (0, 1], got {fraction_to_rotate}.')
  _, _, num_heads, dim_per_head = x.shape

  def _to_even(x):
    return math.floor(x / 2.) * 2
  num_rotated_channels = _to_even(dim_per_head * fraction_to_rotate)
  num_rotated_heads = math.floor(fraction_heads_to_rotate * num_heads)

  if num_rotated_heads != num_heads:
    x_unrotated = x[..., num_rotated_heads:, :]
    x = x[..., :num_rotated_heads, :]

  if num_rotated_channels == dim_per_head:
    x = apply_rotary_encoding(x, max_wavelength, positions)
  else:
    x_r = x[..., :num_rotated_channels]
    x_p = x[..., num_rotated_channels:]
    x_r = apply_rotary_encoding(x_r, max_wavelength, positions)
    x = torch.concat((x_r, x_p), axis=-1)

  if num_rotated_heads != num_heads:
    x = torch.concatenate((x, x_unrotated), axis=-2)
  return x


class Embedding(torch.nn.Module):
    def __init__(self,
                 n_symbols,
                 output_dim,
                 random_state=None,
                 scale=1.,
                 dtype="default",
                 device="default"):
        """
        Last dimension of indices tensor must be 1!!!!
        """
        super(Embedding, self).__init__()
        if random_state is None:
            raise ValueError("Must pass random_state argument to Embedding")

        th_embed = torch.nn.Embedding(n_symbols, output_dim)
        self.th_embed = th_embed
        self.has_warned = False

    def forward(self,
                indices):
        ii = indices.long()
        shp = ii.shape
        nd = len(ii.shape)
        if shp[-1] != 1:
          if nd < 3:
              if self.has_warned == False:
                  print("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                  self.has_warned = True
              ii = ii[..., None]
          else:
              raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = ii.shape
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = self.th_embed(ii[..., 0])
        return lu, self.th_embed.weight

class EmbeddingDropout(Embedding):
    """
    From ENAS
    https://github.com/carpedm20/ENAS-pytorch/blob/master/models/shared_rnn.py
    Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.
    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).
    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """
    def __init__(self,
                 n_symbols,
                 output_dim,
                 dropout_keep_prob=1.,
                 dropout_scale="default",
                 random_state=None,
                 scale=1.,
                 dtype="default",
                 device="default"):
        """Embedding constructor.
        Args:
            dropout_keep_prob: Dropout probability.
            dropout_scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/dropout_keep_prob scaling.
        See `Embedding` for remaining arguments.
        """
        Embedding.__init__(self,
                           n_symbols=n_symbols,
                           output_dim=output_dim,
                           random_state=random_state,
                           scale=scale,
                           dtype=dtype,
                           device=device)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))
        self.device = device

        self.dropout_keep_prob = dropout_keep_prob
        if dropout_scale == "default":
            dropout_scale = output_dim ** 0.5
        self.dropout_scale = dropout_scale
        self.has_warned = False

    def forward(self, indices):
        """Embeds `indices` with the dropped out embedding weight matrix."""

        if self.training:
            dropout_keep_prob = self.dropout_keep_prob
        else:
            dropout_keep_prob = 1.

        if dropout_keep_prob != 1.:
            mask = self.th_embed.weight.data.new(self.th_embed.weight.size(0), 1)
            mask.bernoulli_(dropout_keep_prob, generator=self.g)
            mask = mask.expand_as(self.th_embed.weight)
            mask = mask / (dropout_keep_prob)
            masked_weight = self.th_embed.weight * torch.tensor(mask)
        else:
            masked_weight = self.th_embed.weight

        if self.dropout_scale and self.dropout_scale != 1.:
            masked_weight = masked_weight * self.dropout_scale

        ii = indices.long()
        shp = ii.shape
        nd = len(ii.shape)
        if shp[-1] != 1:
            if nd < 3:
                if self.has_warned == False:
                    print("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                    self.has_warned = True
                ii = ii[..., None]
            else:
                raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = ii.shape
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = nn.functional.embedding(ii[..., 0], masked_weight)
        return lu, masked_weight


class LayerNorm(torch.nn.Module):
    """
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self,
                 input_dim,
                 eps=1E-6,
                 dtype="default",
                 device="default"):
        super(LayerNorm, self).__init__()
        
        self.input_dim = input_dim
        self.a_2 = nn.Parameter(torch.ones(input_dim))
        self.b_2 = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # eps trick wont work here - std has issues if every element is 0 on that axis
        # std = x.std(-1, keepdim=True)
        # want to use sqrt of var + eps instead
        var = x.var(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2
        """
        rms_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed
        """


class Dropout(nn.Module):
    def __init__(self, dropout_keep_prob=1.,
                 random_state=None,
                 dtype="default",
                 device="default"):
        super(Dropout, self).__init__()
        self.dropout = 1. - dropout_keep_prob
        if random_state is None:
            raise ValueError("Must pass random_state to LockedDropout")
        if device == "default":
            raise ValueError("Must pass device argument")
        device = torch.device(device)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))
        self.device = device

    def forward(self, x):
        if not self.training or self.dropout == 0.:
            return x

        pm = x.data.new(*x.size()).zero_()
        pm = 0. * pm + (1. - self.dropout)
        m = torch.bernoulli(pm, generator=self.g)
        mask = m.clone().detach().requires_grad_(False) / (1. - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class LockedDropout(nn.Module):
    def __init__(self, dropout_keep_prob=1.,
                 random_state=None,
                 dtype="default",
                 device="default"):
        super(LockedDropout, self).__init__()
        self.dropout = 1. - dropout_keep_prob
        if random_state is None:
            raise ValueError("Must pass random_state to LockedDropout")
        if device == "default":
            raise ValueError("Must pass device argument")
        device = torch.device(device)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(random_state.randint(100000))
        self.device = device

    def forward(self, x):
        if not self.training or self.dropout == 0.:
            return x
        # assumes data is T B F
        pm = x.data.new(1, *x.size()[1:]).zero_()
        pm = 0. * pm + (1. - self.dropout)
        m = torch.bernoulli(pm, generator=self.g)
        mask = m.clone().detach().requires_grad_(False) / (1. - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class PositionwiseFeedforward(nn.Module):
    def __init__(self,
                 input_dim,
                 projection_dim,
                 dropout_keep_prob=1.0,
                 random_state=None,
                 device="default",
                 dtype="default"):
        super(PositionwiseFeedforward, self).__init__()

        if random_state is None:
            raise ValueError("Must pass random_state to PositionwiseFeedforward")

        self.i = nn.Linear(input_dim,
                           projection_dim,
                           bias=True,
                           device=device,
                           dtype=dtype)

        self.o = nn.Linear(projection_dim,
                           input_dim,
                           bias=True,
                           device=device,
                           dtype=dtype)

        self.ld1 = LockedDropout(dropout_keep_prob=dropout_keep_prob,
                                 device=device,
                                 random_state=random_state)
        self.ld2 = LockedDropout(dropout_keep_prob=dropout_keep_prob,
                                 device=device,
                                 random_state=random_state)

    def forward(self, inp):
        s1 = nn.functional.relu(self.i(inp)) ** 2
        ds1 = self.ld1(s1)

        s2 = self.o(ds1)
        ds2 = self.ld2(s2)
        return ds2


class CrossAttention(torch.nn.Module):
    # assume rotary, query skip, etc from defaults
    def __init__(self,
                 input_q_dim,
                 input_kv_dim,
                 fraction_to_rotate,
                 fraction_heads_to_rotate,
                 n_heads=1,
                 dropout_keep_prob=1.0,
                 attention_dropout_keep_prob=1.0,
                 random_state=None,
                 scale=1.,
                 dtype="default",
                 device="default"):
        super(CrossAttention, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        self.attention_dropout_keep_prob = attention_dropout_keep_prob
        self.attn = Attention(input_q_dim,
                              input_kv_dim,
                              fraction_to_rotate,
                              fraction_heads_to_rotate,
                              n_heads,
                              dropout_keep_prob=attention_dropout_keep_prob,
                              random_state=random_state,
                              device=device,
                              dtype=dtype)
        self.ln_q = LayerNorm(input_q_dim,
                              device=device,
                              dtype=dtype)
        self.ln_kv = LayerNorm(input_kv_dim,
                               device=device,
                               dtype=dtype)
        self.ln_x = LayerNorm(input_q_dim,
                              device=device,
                              dtype=dtype)
        inner_dropout_keep_prob = dropout_keep_prob
        input_dim = input_q_dim
        inner_dim = int(1.0 * input_dim)
        self.ff = PositionwiseFeedforward(input_dim,
                                          inner_dim,
                                          dropout_keep_prob=inner_dropout_keep_prob,
                                          random_state=random_state,
                                          device=device,
                                          dtype=dtype)
        self.drop = Dropout(dropout_keep_prob=self.dropout_keep_prob,
                            device=device,
                            random_state=random_state)

    def forward(self,
                inputs_q,
                inputs_kv,
                mask,
                q_positions,
                kv_positions,):
        attn, attn_state = self.attn(self.ln_q(inputs_q),
                                     self.ln_kv(inputs_kv),
                                     mask=mask,
                                     q_positions=q_positions,
                                     kv_positions=kv_positions,
                                     is_cross_attend=True)
        attn = self.drop(attn)
        x = inputs_q + attn
        x = x + self.ff(self.ln_x(x))
        return x, attn_state


class SelfAttention(torch.nn.Module):
    # assume rotary, query skip, etc from defaults
    def __init__(self,
                 input_dim,
                 fraction_to_rotate,
                 fraction_heads_to_rotate,
                 inner_dim=None,
                 n_heads=8,
                 dropout_keep_prob=1.0,
                 attention_dropout_keep_prob=1.0,
                 random_state=None,
                 scale=1.,
                 dtype="default",
                 device="default"):
        super(SelfAttention, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        self.attention_dropout_keep_prob = attention_dropout_keep_prob
        self.attn = Attention(input_dim,
                              input_dim,
                              fraction_to_rotate,
                              fraction_heads_to_rotate,
                              n_heads,
                              dropout_keep_prob=attention_dropout_keep_prob,
                              random_state=random_state,
                              device=device,
                              dtype=dtype)
        self.ln = LayerNorm(input_dim,
                            device=device,
                            dtype=dtype)
        self.ln_x = LayerNorm(input_dim,
                              device=device,
                              dtype=dtype)
        inner_dropout_keep_prob = dropout_keep_prob
        if inner_dim is None:
            inner_dim = int(4.0 * input_dim)
        self.ff = PositionwiseFeedforward(input_dim,
                                          inner_dim,
                                          dropout_keep_prob=inner_dropout_keep_prob,
                                          random_state=random_state,
                                          device=device,
                                          dtype=dtype)
        self.drop = Dropout(dropout_keep_prob=self.dropout_keep_prob,
                            device=device,
                            random_state=random_state)

    def forward(self,
                inputs_qkv,
                mask,
                q_positions,
                kv_positions,):
        qkv = self.ln(inputs_qkv)
        attn, attn_state = self.attn(qkv,
                                     qkv,
                                     mask=mask,
                                     q_positions=q_positions,
                                     kv_positions=kv_positions,
                                     is_cross_attend=False)
        x = inputs_qkv + self.drop(attn)
        x = x + self.ff(self.ln_x(x))
        return x, attn_state


class Attention(torch.nn.Module):
    # assume rotary, query skip, etc from defaults
    def __init__(self,
                 input_q_dim,
                 input_kv_dim,
                 fraction_to_rotate,
                 fraction_heads_to_rotate,
                 n_heads=1,
                 dropout_keep_prob=1.0,
                 max_rotary_wavelength=8192,
                 random_state=None,
                 scale=1.,
                 dtype="default",
                 device="default"):
        super(Attention, self).__init__()
        self.device = device
        self.dtype = dtype
        self.n_heads = n_heads
        assert input_q_dim % n_heads == 0
        assert input_kv_dim % n_heads == 0
        self.qk_channels = input_q_dim
        self.v_channels = input_kv_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.qk_channels_per_head = input_q_dim // self.n_heads
        self.v_channels_per_head = input_kv_dim // self.n_heads
        self.fraction_to_rotate = fraction_to_rotate
        self.fraction_heads_to_rotate = fraction_heads_to_rotate
        self.max_rotary_wavelength = max_rotary_wavelength
        self.q_head = nn.Linear(input_q_dim,
                                input_q_dim,
                                bias=True,
                                device=self.device,
                                dtype=self.dtype)
        
        self.k_head = nn.Linear(input_kv_dim,
                                input_q_dim,
                                bias=True,
                                device=self.device,
                                dtype=self.dtype)
        
        self.v_head = nn.Linear(input_kv_dim,
                                input_q_dim,
                                bias=True,
                                device=self.device,
                                dtype=self.dtype)
        
        self.out = nn.Linear(input_q_dim,
                             input_q_dim,
                             bias=True,
                             device=self.device,
                             dtype=self.dtype)
        self.drop = Dropout(dropout_keep_prob=self.dropout_keep_prob,
                            device=device,
                            random_state=random_state)

        
    def _rotary_position_embeddings(self,
                                    q,
                                    k,
                                    q_positions,
                                    kv_positions):
        head_dim = q.shape[-1]
        rotary_queries = apply_rotary_encoding_to_subset(
                           q, self.fraction_to_rotate, self.fraction_heads_to_rotate,
                           self.max_rotary_wavelength, q_positions)
        rotary_keys = apply_rotary_encoding_to_subset(
                        k, self.fraction_to_rotate, self.fraction_heads_to_rotate,
                        self.max_rotary_wavelength, kv_positions)
        return rotary_queries, rotary_keys

    def _attend(self,
                q,
                k,
                v,
                mask,
                q_positions,
                kv_positions,
                is_cross_attend):
            q_indices, batch, num_heads, head_dim = q.shape
            hiddens = num_heads * head_dim
            rotary_queries, rotary_keys = self._rotary_position_embeddings(q, k, q_positions, kv_positions)
            attention = torch.einsum('tbhd,Tbhd->hbtT', rotary_queries, rotary_keys)
            # attention now head batch latent T
            scale = 1. / math.sqrt(head_dim)
            attention = attention * scale
            # T B latent -> add broadcast for heads 1 T B latent
            mask = mask[None]
            # 1 T B latent -> head B latent T
            mask = mask.permute(0, 2, 3, 1)
            attention = torch.where(mask > 0, attention, 0. * attention + -1E30)
            # mask values of 0 indicate entry will be masked
            normalized = torch.nn.functional.softmax(attention, dim=-1)
            # latent batch head dim
            normalized_drop = self.drop(normalized)
            summed = torch.einsum('hbtT,Tbhd->tbhd', normalized_drop, v)
            return summed.reshape((q_indices, batch, hiddens))

    def forward(self,
                inputs_q,
                inputs_kv,
                mask,
                q_positions,
                kv_positions,
                is_cross_attend):
        # do as 1 or as 3?
        q = self.q_head(inputs_q)
        k = self.k_head(inputs_kv)
        v = self.v_head(inputs_kv)
        qt_, b_, qf_ = q.shape
        kt_, b_, kf_ = k.shape
        vt_, b_, vf_ = v.shape

        q = q.reshape((qt_, b_, self.n_heads, self.qk_channels_per_head))
        k = k.reshape((kt_, b_, self.n_heads, self.qk_channels_per_head))
        v = v.reshape((vt_, b_, self.n_heads, self.v_channels_per_head))

        result = self._attend(q, k, v,
                              mask=mask,
                              q_positions=q_positions,
                              kv_positions=kv_positions,
                              is_cross_attend=is_cross_attend)

        out = self.out(result)
        return out, None


class PerceiverSUNMASK(nn.Module):
    def __init__(self,
                 n_classes,
                 z_index_dim,
                 n_processor_layers,
                 input_embed_dim=1024,
                 num_z_channels=1024,
                 inner_expansion_dim=4096,
                 input_dropout_keep_prob=1.0,
                 cross_attend_dropout_keep_prob=1.0,
                 autoregression_dropout_keep_prob=1.0,
                 inner_dropout_keep_prob=1.0,
                 final_dropout_keep_prob=1.0,
                 learnable_position_embeddings=False,
                 position_encoding_type='rotary',
                 fraction_to_rotate=0.25,
                 fraction_heads_to_rotate=1.0):
        super(PerceiverSUNMASK, self).__init__()
        self.dtype = get_dtype_default()
        self.device = get_device_default()
        self._n_classes = n_classes
        self._input_embed_dim = input_embed_dim
        self._num_z_channels = num_z_channels
        self._inner_expansion_dim = inner_expansion_dim
        self._z_index_dim = z_index_dim
        self._n_processor_layers = n_processor_layers
        self._position_encoding_type = position_encoding_type
        self._fraction_to_rotate = fraction_to_rotate
        self._fraction_heads_to_rotate = fraction_heads_to_rotate

        self._input_dropout_keep_prob = input_dropout_keep_prob
        self._cross_attend_dropout_keep_prob = cross_attend_dropout_keep_prob
        self._autoregression_dropout_keep_prob = autoregression_dropout_keep_prob
        self._inner_dropout_keep_prob = inner_dropout_keep_prob
        self._final_dropout_keep_prob = final_dropout_keep_prob

        self._init_random_state = np.random.RandomState(1234)
        self._init_generator_seed = self._init_random_state.randint(100000)

        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(self._init_generator_seed)

        # input has +1 due to shift/offset of 1 in batch processing (to allow for 0 to denote masked length)
        self.input_embed = EmbeddingDropout(self._n_classes + 1 + 1, self._input_embed_dim,
                                     dropout_keep_prob=self._input_dropout_keep_prob,
                                     random_state=self._init_random_state,
                                     device=self.device,
                                     dtype=self.dtype)
        # div 2 because of sunmask
        self.input_query_embed = Embedding(self._n_classes + 1 + 1, self._input_embed_dim // 2,
                                     random_state=self._init_random_state,
                                     device=self.device,
                                     dtype=self.dtype)
        self.z_linear = nn.Linear(self._input_embed_dim,
                                  self._num_z_channels,
                                  bias=True,
                                  device=self.device,
                                  dtype=self.dtype)
        self.ln_o = LayerNorm(self._num_z_channels,
                              device=self.device,
                              dtype=self.dtype)

        self.final_layer = nn.Linear(self._num_z_channels,
                                     self._n_classes,
                                     bias=True,
                                     device=self.device,
                                     dtype=self.dtype)

        self.drop = Dropout(dropout_keep_prob=self._final_dropout_keep_prob,
                            device=self.device,
                            random_state=self._init_random_state)

        self.initial_cross_attn = CrossAttention(self._num_z_channels,
                                                 self._input_embed_dim,
                                                 fraction_to_rotate=.25,
                                                 fraction_heads_to_rotate=1.0,
                                                 n_heads=1,
                                                 dropout_keep_prob=self._inner_dropout_keep_prob,
                                                 attention_dropout_keep_prob=self._inner_dropout_keep_prob,
                                                 random_state=self._init_random_state,
                                                 device=self.device,
                                                 dtype=self.dtype)
        self.processor_layers = nn.ModuleList()
        for _i in range(self._n_processor_layers):
            self_attn = SelfAttention(self._num_z_channels,
                                      fraction_to_rotate=.25,
                                      fraction_heads_to_rotate=1.0,
                                      n_heads=10,
                                      inner_dim=self._inner_expansion_dim,
                                      dropout_keep_prob=self._inner_dropout_keep_prob,
                                      attention_dropout_keep_prob=self._inner_dropout_keep_prob,
                                      random_state=self._init_random_state,
                                      device=self.device,
                                      dtype=self.dtype)
            self.processor_layers.append(self_attn)

    def reset_generator(self):
        self._init_random_state = np.random.RandomState(1234)
        self.g.manual_seed(self._init_generator_seed)

    def _build_network_inputs(self, inputs, input_idxs):
        assert len(inputs.shape) == 2
        # time, batch format
        if self._position_encoding_type != "rotary":
            raise AttributeError("Only support rotary encoding at present")
        assert input_idxs is not None
        assert input_idxs.shape[-1] == 1
        positions = input_idxs[..., 0]
        embedded_inputs_for_input = self.input_embed(inputs)
        embedded_inputs_for_query = self.input_query_embed(inputs)
        # mask embedding directly? or no
        return embedded_inputs_for_input[0], embedded_inputs_for_query[0], positions

    def forward(self, inputs, input_idxs, input_mask):
        batch_size = inputs.shape[1]
        z_index_dim = self._z_index_dim
        # masks[0] encoder, masks[1] processor
        # time batch latent for encoder, latent batch latent for processor
        masks, latent_last_steps = make_block_causal_masks(inputs,
                                                           latent_index_dim=z_index_dim,
                                                           batch_size=batch_size)
        # make encoder mask allow all
        # these masks are 1 -> allow, 0 drop
        new_masks = [masks[0], masks[1]]
        new_masks[0] = (0. * masks[0] + 1.).type(masks[0].dtype)
        # make processor mask allow all
        new_masks[1] = (0. * masks[1] + 1.).type(masks[1].dtype)
        masks = new_masks

        # time batch input_embed_dim
        embedded_inputs_inputs, embedded_inputs_query, positions = self._build_network_inputs(inputs, input_idxs)
        assert self._position_encoding_type == "rotary"

        # latent, batch
        latent_positions = torch.gather(positions, 0, latent_last_steps.T.long())

        # latent batch embedding dim
        last_step_embeddings = torch.gather(embedded_inputs_query, 0, latent_last_steps.T[..., None].repeat(1, 1, self._input_embed_dim // 2).long())
        # gathered the last z_index_dim elements of each sequence (before 0 pad), positions and their embed
        # latent_T B feat
        # not really embed just scale and stretch
        input_mask_embed = input_mask[..., None] * 1. / np.sqrt(self._input_embed_dim)
        input_mask_embed = 0. * last_step_embeddings + input_mask_embed
        last_step_embeddings = torch.concat([last_step_embeddings, input_mask_embed], axis=-1)

        initial_q_input = self.z_linear(last_step_embeddings)
        # can we just do cross attn dropout here?

        memory = None
        if memory is None:
          attn_memory = None

        # do cross attend dropout here
        # drop elements out of encoder cross attend mask
        if self.training and self._cross_attend_dropout_keep_prob != 1.0:
            pm = masks[0].data.new(*masks[0].size()).zero_()
            pm = 0. * pm + self._cross_attend_dropout_keep_prob
            m = torch.bernoulli(pm, generator=self.g)
            # coin flip whether to keep or drop every element independently
            drop_encoder_mask = (m * masks[0]).type(masks[0].dtype)
        else:
            drop_encoder_mask = masks[0]

        z, initial_cross_attend_state = self.initial_cross_attn(
            initial_q_input, embedded_inputs_inputs,
            mask=drop_encoder_mask,
            # Input position encodings belong to the keys/values, so here we use
            # the inputs' position encodings rather than the latents'.
            q_positions=latent_positions,
            kv_positions=positions)

        # latent, batch, hid
        # do AR drop locked dropout style
        # same mask all layers
        # drop elements out of encoder cross attend mask
        if self.training and self._autoregression_dropout_keep_prob != 1.0:
            pm = masks[1].data.new(*masks[1].size()).zero_()
            pm = 0. * pm + self._autoregression_dropout_keep_prob
            m = torch.bernoulli(pm, generator=self.g)
            # coin flip whether to keep or drop every element independently
            drop_decoder_mask = (m * masks[1]).type(masks[1].dtype)
        else:
            drop_decoder_mask = masks[1]

        for processor_l in self.processor_layers:
            z, initial_cross_attend_state = processor_l(
                z,
                mask=drop_decoder_mask,
                q_positions=latent_positions,
                kv_positions=latent_positions)
        z = self.ln_o(z)
        z = self.drop(z)
        return self.final_layer(z), masks
