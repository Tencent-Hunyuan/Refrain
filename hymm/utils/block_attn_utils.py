import torch
import math

def padding_text_seq_to_block_size(text_emb, block_size):
    """把text_embedding padding到block_size"""
    b, s, h, d = text_emb.shape
    pad_size = 0
    new_seq_len =  block_size * math.ceil(s / block_size) # 向上取整到block_size的整数倍
    pad_size = new_seq_len - s
    if pad_size > 0:
        last_vec = text_emb[:, -1:, :, :]  # 取最后一个token的特征 [b, 1, h, d]
        pad_tensor = last_vec.repeat(1, pad_size, 1, 1)  # 沿序列维度复制
        text_emb = torch.cat((text_emb, pad_tensor), dim=1)
    return text_emb, pad_size


def qkv_to_flash_shape(qkv):
    """把序列变成1D的序列长度, 适配flash attention的输入格式"""
    b, s, a, d = qkv.shape
    return qkv.reshape(b * s, a, d)

def flash_to_ori_shape(o, b):
    """把flash attention的输出变成原始的形状"""
    bs, a, d = o.shape
    s = bs//b
    return o.reshape(b, s, a, d)

def bshd_to_bcthw(tensor, image_shape):
    """把1D的序列变成3D的视频序列维度"""
    T,H,W= image_shape
    b,s,h,d = tensor.shape
    tensor = tensor.view(b, T, H, W, h*d)
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
    return tensor

def batch_slice_4d_tensor(tensor, image_shape, tube_shape, head, dim, use_padding=False):
    T,H,W = image_shape
    tube_t, tube_h, tube_w = tube_shape
    # input : B * C * T * H * W
    # output: B * C * S -> B * S * head * dim
    tensor=bshd_to_bcthw(tensor, image_shape)
    assert tensor.dim() == 5, "Input tensor must be 5D (B * C * T * H * W), got {} \
                                T: {}, H: {}, W: {}".format(tensor.size(), T, H, W)
    slice_tensor = []
    
    # padding T,H,W to ensure they are multiple of tube_t, tube_h, tube_w
    # recored the padding size
    pad_t = 0
    pad_h = 0
    pad_w = 0
    if use_padding:
        if T % tube_t != 0:
            pad_t = tube_t - (T % tube_t)
            pad_tensor = tensor[:, :, -1, :, :].unsqueeze(2).repeat(1, 1, pad_t, 1, 1)
            tensor = torch.cat((tensor, pad_tensor), dim=2)
            T += pad_t
        if H % tube_h!= 0:
            pad_h = tube_h - (H % tube_h)
            pad_tensor = tensor[:, :, :, -1, :].unsqueeze(3).repeat(1, 1, 1, pad_h, 1)  
            tensor = torch.cat((tensor, pad_tensor), dim=3) 
            H += pad_h
        if W % tube_w!= 0:
            pad_w = tube_w - (W % tube_w)
            pad_tensor = tensor[:, :, :, :, -1].unsqueeze(4).repeat(1, 1, 1, 1, pad_w)
            tensor = torch.cat((tensor, pad_tensor), dim=4) 
            W += pad_w  
        
    for i in range(0, T, tube_t):
        for j in range(0, H, tube_h):
            for k in range(0, W, tube_w):
                flat_tensor = torch.flatten(tensor[:, :, i:i+tube_t, j:j+tube_h, k:k+tube_w], start_dim=2)
                slice_tensor.append(flat_tensor)
    slice_tensor = torch.concat(slice_tensor, dim=-1) # B * C * L
    B,C,L = slice_tensor.shape # C=head*dim
    slice_tensor = slice_tensor.view(B, head, dim, L).permute(0, 3, 1, 2).contiguous()
    return slice_tensor, (T, H, W) ,(pad_t, pad_h, pad_w)

def batch_sliced_tensor_to_4d(input, image_shape, tube_shape, pad_image_shape=None, pad_size=None):
    # input : B * Seq * Head * Dim -> B * C(Head * Dim) * T * H * W -> B * Seq * Head * Dim
    B, _, head, dim = input.shape
    T,H,W = pad_image_shape
    tube_t, tube_h, tube_w = tube_shape
    assert input.dim() == 4, "Input tensor must be 4D (B * S * H * D), got {}".format(input.size())
    
    output = torch.zeros((B, head*dim, T, H, W), device=input.device, dtype=input.dtype)
    start_idx = 0
    for i in range(0, T, tube_t):
        current_tube_t = min(tube_t, T - i)
        for j in range(0, H, tube_h):
            current_tube_h = min(tube_h, H - j)
            for k in range(0, W, tube_w):
                current_tube_w = min(tube_w, W - k)
                current_length = current_tube_t * current_tube_h * current_tube_w
                current_slice = input[:, start_idx: start_idx + current_length, :, :]
                current_slice = current_slice.permute(0, 2, 3, 1).contiguous()
                current_block = current_slice.view(B, head*dim, current_tube_t, current_tube_h, current_tube_w)
                output[:, :, i:i+current_tube_t, j:j+current_tube_h, k:k+current_tube_w] = current_block
                start_idx += current_length
    if pad_size is not None:
        pad_t, pad_h, pad_w = pad_size
        if pad_t > 0:
            output = output[:, :, :T-pad_t, :, :]
        if pad_h > 0:
            output = output[:, :, :, :H-pad_h, :]
        if pad_w > 0:
            output = output[:, :, :, :, :W-pad_w]
    
    output = output.permute(0, 2, 3, 4, 1).contiguous()
    output = output.view(B, -1, head, dim)
    return output

def test():
    size=(2, 1024, 128, 32) # B * S * H * D
    sample_data = torch.randn(size, dtype=torch.float32, device="cuda")
    print(sample_data.shape)
    tube_shape=(1, 9, 9)
    output, (T, H, W) ,(pad_t, pad_h, pad_w) = batch_slice_4d_tensor(sample_data, (1,32,32),tube_shape, 128, 32)
    print(output.shape)
    output = batch_sliced_tensor_to_4d(output, (1, 32,32), tube_shape, (T, H, W), (pad_t, pad_h, pad_w))
    print(output.shape)
    equal = torch.equal(sample_data, output)
    print(sample_data[0,0,0,0:10])
    print(output[0,0,0,0:10])
    if equal:
        print("The original data and the reconstructed data are equal.")
    else:
        print("The original data and the reconstructed data are not equal.")
