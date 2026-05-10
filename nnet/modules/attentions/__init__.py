from .multi_head_attention import MultiHeadAttention
from .rel_pos_multi_head_self_attention import RelPosMultiHeadSelfAttention
from .attention_mask import return_mask, return_padding_mask, return_is_firsts_mask

# Attentions Dictionary
att_dict = {
    "MultiHeadAttention": MultiHeadAttention,
    "RelPosMultiHeadSelfAttention": RelPosMultiHeadSelfAttention
}