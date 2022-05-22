import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

import copy
import math
import numpy as np
import random

def bpr_loss(pos_score, neg_score, gamma=1e-10):
    loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score))
    loss = loss.mean()
    return loss


class Dice(nn.Module):
    r"""Dice activation function
    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s
    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


# Transformer
class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask if attention_mask != None else attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": nn.ReLU,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

# NextitNet
class ResidualBlock_a(nn.Module):
    r"""
    Residual block (a) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()

        half_channel = out_channel // 2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channel, half_channel, kernel_size=(1, 1), padding=0)

        self.ln2 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)

        self.ln3 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv3 = nn.Conv2d(half_channel, out_channel, kernel_size=(1, 1), padding=0)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]

        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)

        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)

        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):  # x: [batch_size, seq_len, embed_size]
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        # padding operation  args：(left,right,top,bottom)
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.randn(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.randn(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.randn(self.gate_size))
        self.b_hh = nn.Parameter(torch.randn(self.gate_size))
        self.b_iah = nn.Parameter(torch.randn(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.randn(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.
        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]
            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]
        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]
        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

# Ada-Ranker
class NeuProcessEncoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64, dropout_prob=0.4, device=None):
        super(NeuProcessEncoder, self).__init__()
        self.device = device
        
        # Encoder for item embeddings
        layers = [nn.Linear(input_size, hidden_size),
                torch.nn.Dropout(dropout_prob),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, output_size)]
        self.input_to_hidden = nn.Sequential(*layers)

        # Encoder for latent vector z
        self.z1_dim = input_size # 64
        self.z2_dim = hidden_size # 64
        self.z_dim = output_size # 64
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def emb_encode(self, input_tensor):
        hidden = self.input_to_hidden(input_tensor)

        return hidden

    def aggregate(self, input_tensor):
        return torch.mean(input_tensor, dim=-2)

    def z_encode(self, input_tensor):
        hidden = torch.relu(self.z_to_hidden(input_tensor))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, log_sigma
    
    def encoder(self, input_tensor):
        z_ = self.emb_encode(input_tensor)
        z = self.aggregate(z_)
        self.z, mu, log_sigma = self.z_encode(z)
        return self.z, mu, log_sigma

    def forward(self, input_tensor):
        self.z, _, _ = self.encoder(input_tensor)
        return self.z


class AdaLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight_new.transpose(-1, -2))

        output = input.matmul(self.weight_new.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias_new
        return output

    def adaptive_parameters(self, batch_size, gama, beta):
        '''
        gama: [batch_size, self.out_features, self.in_features]
        beta: [batch_size, 1]
        self.weight.data: [self.out_features, self.in_features]
        '''
        gama_w = gama.unsqueeze(1).expand([batch_size, self.out_features, self.in_features])
        beta_w = beta.unsqueeze(1)
        gama_b = gama.expand([batch_size, self.out_features])
        beta_b = beta

        self.weight_specific = self.weight * gama_w + beta_w # [batch_size, self.out_features, self.in_features]
        self.weight_new = self.weight_specific * self.weight

        if self.bias is not None:
            self.bias_specific = self.bias * gama_b + beta_b
            self.bias_new = self.bias_specific + self.bias
            self.bias_new = self.bias_new.unsqueeze(1)
    
    def adaptive_parameters_ws(self, batch_size, gama, beta):
        '''
        gama: [batch_size, self.out_features, self.in_features]
        beta: [batch_size, 1]
        self.weight.data: [self.out_features, self.in_features]
        '''
        gama_w = gama.unsqueeze(1).expand([batch_size, self.out_features, self.in_features])
        beta_w = beta.unsqueeze(1)
        gama_b = gama.expand([batch_size, self.out_features])
        beta_b = beta

        self.weight_new = self.weight * gama_w + beta_w # [batch_size, self.out_features, self.in_features]

        if self.bias is not None:
            self.bias_new = self.bias * gama_b + beta_b
            self.bias_new = self.bias_new.unsqueeze(1)
    
    def memory_parameters(self, mem_wei, mem_bias):
        self.weight_specific = mem_wei # [batch_size, self.out_features, self.in_features]
        self.weight_new = self.weight_specific * self.weight

        if self.bias is not None:
            self.bias_specific = mem_bias.squeeze(-1)
            self.bias_new = self.bias_specific + self.bias
            self.bias_new = self.bias_new.unsqueeze(1)
    
    def add_bias_only(self, bias_vec):
        self.weight_new = self.weight
        self.bias_new = bias_vec + self.bias
        self.bias_new = self.bias_new.unsqueeze(1)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, clusters_k=10):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.input_size = input_size
        self.output_size = output_size
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, input_size*output_size)))
        self.index = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, emb_size)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        att_scores = torch.matmul(bias_emb, self.index.transpose(-1, -2)) # [batch_size, clusters_k]
        att_scores = self.softmax(att_scores)

        # [batch_size, input_size, output_size]
        para_new = torch.matmul(att_scores, self.array) # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size)

        return para_new

    def reg_loss(self, reg_weights=1e-2):
        loss_1 = reg_weights * self.array.norm(2)
        loss_2 = reg_weights * self.index.norm(2)

        return loss_1 + loss_2


class ModulateHidden(nn.Module):
    def __init__(self, input_size, emb_size):
        super(ModulateHidden, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.gen_para_layer = nn.Linear(self.emb_size, self.input_size*self.input_size)

    def gen_para(self, bias_emb):
        """
        bias_emb: [batch_size, emb_size]
        """
        para_new = self.gen_para_layer(bias_emb) # [batch_size, self.input_size*self.output_size]
        self.para_new = para_new.view(-1, self.input_size, self.input_size)

    def forward(self, input: torch.Tensor):
        output = input.matmul(self.para_new.transpose(-1, -2))
        
        return output

class AdapLinear_mmoe(nn.Module):
    
    def __init__(self, config, emb_size, in_features: int, out_features: int, bias: bool = True, expert_num=10, device=None, dtype=None):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdapLinear_mmoe, self).__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        _weight = torch.randn((expert_num, self.out_features*self.in_features), requires_grad=True).to(self.device)
        self.weight = nn.Parameter(_weight)

        if bias:
            _bias = torch.randn((expert_num, self.out_features), requires_grad=True).to(self.device)
            self.bias = nn.Parameter(_bias)
            # self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))

        # gate
        self.gate_net = nn.Linear(emb_size, expert_num, bias=False)
        self.softmax = nn.Softmax(-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight_new.transpose(-1, -2))

        output = input.matmul(self.weight_new.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias_new
        return output

    def adaptive_parameters(self, domain_bias):
        # domain_bias: [batch_size, emb_size]
        if len(domain_bias.size()) == 3:
            domain_bias = domain_bias.squeeze(1)
        att_scores = self.gate_net(domain_bias) # [batch_size, expert_num]
        att_scores = self.softmax(att_scores)
        self.weight_new = torch.matmul(att_scores, self.weight) # [batch_size, input_size*output_size]
        self.weight_new = self.weight_new.view(-1, self.out_features, self.in_features) # [batch_size, self.out_features, self.in_features]
        if self.bias is not None:
            self.bias_new = torch.matmul(att_scores, self.bias).unsqueeze(1) # [batch_size, input_size*output_size]
            
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MMoEUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, expert_num=10):
        super(MMoEUnit, self).__init__()
        self.expert_num = expert_num
        self.input_size = input_size
        self.output_size = output_size

        _weight = torch.randn((expert_num, self.output_size*self.input_size), requires_grad=True)
        self.weight = nn.Parameter(_weight)

        # gate
        self.gate_net = nn.Linear(emb_size, expert_num, bias=False)
        self.softmax = nn.Softmax(-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        if len(bias_emb.size()) == 3:
            bias_emb = bias_emb.squeeze(1)
        att_scores = self.gate_net(bias_emb) # [batch_size, expert_num]
        att_scores = self.softmax(att_scores)
        para_new = torch.matmul(att_scores, self.weight) # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size) # [batch_size, self.out_features, self.in_features]

        return para_new
