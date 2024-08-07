import torch

dim = 6
head = 2
layer_num = 6

encoder_seq = 2
encoder_batch_size = 2
encoder_input = torch.rand(encoder_batch_size,encoder_seq,dim)

decoder_seq = 2
decoder_batch_size = 2
decoder_input = torch.rand(decoder_batch_size,decoder_seq,dim)

transformer_model = torch.nn.Transformer(d_model=dim,
                nhead=head,num_encoder_layers=layer_num,
                num_decoder_layers=layer_num,
                dropout=0.0,batch_first=True)
out = transformer_model(encoder_input,decoder_input)
print(out)

def my_scaled_dot_product(query,key,value):
    qk_T = torch.mm(query,key.T)
    qk_T_scale = qk_T / torch.sqrt(torch.tensor(value.shape[1]))
    qk_exp = torch.exp(qk_T_scale)
    qk_exp_sum = torch.sum(qk_exp,dim=1,keepdim=True)
    qk_softmax = qk_exp / qk_exp_sum
    v_attn = torch.mm(qk_softmax,value)
    return v_attn,qk_softmax

def my_encoder_layer(encoder_layer,value,index):
    in_proj_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.self_attn.in_proj_weight']
    in_proj_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.self_attn.in_proj_bias']
    out_proj_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.self_attn.out_proj.weight']
    out_proj_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.self_attn.out_proj.bias']
    batch_V_output = torch.empty(encoder_batch_size,encoder_seq,dim)
    for i in range(encoder_batch_size):
        in_proj = torch.mm(value[i],in_proj_weight.T) + in_proj_bias
        Qs,Ks,Vs = torch.split(in_proj,dim,dim=-1)
        head_Vs = []
        for Q,K,V in zip(torch.split(Qs,dim//head,dim=-1),torch.split(Ks,dim//head,dim=-1),torch.split(Vs,dim//head,dim=-1)):
            head_v,_ = my_scaled_dot_product(Q,K,V)
            head_Vs.append(head_v)
        V_cat = torch.cat(head_Vs,dim=-1)
        V_ouput = torch.mm(V_cat,out_proj_weight.T) + out_proj_bias
        batch_V_output[i] = V_ouput
    # 第一次加
    first_Add = value + batch_V_output
    # 第一次layer_norm
    norm1_mean = torch.mean(first_Add,dim=-1,keepdim=True)
    norm1_std = torch.sqrt(torch.var(first_Add,unbiased=False,dim=-1,keepdim=True) + 1e-5)
    norm1_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.norm1.weight']
    norm1_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.norm1.bias']
    norm1 = ((first_Add - norm1_mean)/norm1_std) * norm1_weight + norm1_bias
    # feed forward
    linear1_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.linear1.weight']
    linear1_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.linear1.bias']
    linear2_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.linear2.weight']
    linear2_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.linear2.bias']
    linear1 = torch.matmul(norm1,linear1_weight.T) + linear1_bias
    linear1_relu = torch.nn.functional.relu(linear1)
    linear2 = torch.matmul(linear1_relu,linear2_weight.T) + linear2_bias
    # 第二次加
    second_Add = norm1 + linear2
    # 第二次layer_norm
    norm2_mean = torch.mean(second_Add,dim=-1,keepdim=True)
    norm2_std = torch.sqrt(torch.var(second_Add,unbiased=False,dim=-1,keepdim=True) + 1e-5)
    norm2_weight = encoder_layer.state_dict()[f'encoder.layers.{index}.norm2.weight']
    norm2_bias = encoder_layer.state_dict()[f'encoder.layers.{index}.norm2.bias']
    norm2 = ((second_Add - norm2_mean)/norm2_std) * norm2_weight + norm2_bias
    return norm2

encoder_out = torch.empty_like(encoder_input)
encoder_out[:] = encoder_input
for i in range(layer_num):
    encoder_out = my_encoder_layer(transformer_model,encoder_out,i)
encoder_mean = torch.mean(encoder_out,dim=-1,keepdim=True)
encoder_std = torch.sqrt(torch.var(encoder_out,unbiased=False,dim=-1,keepdim=True) + 1e-5)
encoder_weight = transformer_model.state_dict()['encoder.norm.weight']
encoder_bias = transformer_model.state_dict()['encoder.norm.bias']
encoder_out = ((encoder_out - encoder_mean)/encoder_std) * encoder_weight + encoder_bias

def my_decoder_layer(decoder_layer,decoder_input,encoder_output,index):
    # 第一次multi-head
    first_in_proj_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.self_attn.in_proj_weight']
    first_in_proj_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.self_attn.in_proj_bias']
    first_out_proj_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.self_attn.out_proj.weight']
    first_out_proj_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.self_attn.out_proj.bias']
    first_batch_V_output = torch.empty(decoder_batch_size,decoder_seq,dim)
    for i in range(decoder_batch_size):
        first_in_proj = torch.mm(decoder_input[i],first_in_proj_weight.T) + first_in_proj_bias
        Qs,Ks,Vs = torch.split(first_in_proj,dim,dim=-1)
        head_Vs = []
        for Q,K,V in zip(torch.split(Qs,dim//head,dim=-1),torch.split(Ks,dim//head,dim=-1),torch.split(Vs,dim//head,dim=-1)):
            head_v,_ = my_scaled_dot_product(Q,K,V)
            head_Vs.append(head_v)
        V_cat = torch.cat(head_Vs,dim=-1)
        V_ouput = torch.mm(V_cat,first_out_proj_weight.T) + first_out_proj_bias
        first_batch_V_output[i] = V_ouput
    # 第一次加
    first_Add = decoder_input + first_batch_V_output
    # 第一次layer_norm
    norm1_mean = torch.mean(first_Add,dim=-1,keepdim=True)
    norm1_std = torch.sqrt(torch.var(first_Add,unbiased=False,dim=-1,keepdim=True) + 1e-5)
    norm1_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.norm1.weight']
    norm1_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.norm1.bias']
    norm1 = ((first_Add - norm1_mean)/norm1_std) * norm1_weight + norm1_bias
    # 第二次multi-head
    second_in_proj_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.multihead_attn.in_proj_weight']
    second_in_proj_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.multihead_attn.in_proj_bias']
    second_out_proj_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.multihead_attn.out_proj.weight']
    second_out_proj_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.multihead_attn.out_proj.bias']
    second_batch_V_output = torch.empty(decoder_batch_size,decoder_seq,dim)
    for i in range(decoder_batch_size):
        Qs_weight,Ks_weight,Vs_weight = torch.split(second_in_proj_weight.T,dim,dim=-1)
        Qs_bias,Ks_bias,Vs_bias = torch.split(second_in_proj_bias,dim,dim=-1)
        Qs = torch.mm(norm1[i],Qs_weight) + Qs_bias
        Ks = torch.mm(encoder_output[i],Ks_weight) + Ks_bias
        Vs = torch.mm(encoder_output[i],Vs_weight) + Vs_bias
        head_Vs = []
        for Q,K,V in zip(torch.split(Qs,dim//head,dim=-1),torch.split(Ks,dim//head,dim=-1),torch.split(Vs,dim//head,dim=-1)):
            head_v,_ = my_scaled_dot_product(Q,K,V)
            head_Vs.append(head_v)
        V_cat = torch.cat(head_Vs,dim=-1)
        V_ouput = torch.mm(V_cat,second_out_proj_weight.T) + second_out_proj_bias
        second_batch_V_output[i] = V_ouput
    # 第二次加
    second_Add = norm1 + second_batch_V_output
    # 第二次layer_norm
    norm2_mean = torch.mean(second_Add,dim=-1,keepdim=True)
    norm2_std = torch.sqrt(torch.var(second_Add,unbiased=False,dim=-1,keepdim=True) + 1e-5)
    norm2_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.norm2.weight']
    norm2_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.norm2.bias']
    norm2 = ((second_Add - norm2_mean)/norm2_std) * norm2_weight + norm2_bias
    # feed forward
    linear1_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.linear1.weight']
    linear1_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.linear1.bias']
    linear2_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.linear2.weight']
    linear2_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.linear2.bias']
    linear1 = torch.matmul(norm2,linear1_weight.T) + linear1_bias
    linear1_relu = torch.nn.functional.relu(linear1)
    linear2 = torch.matmul(linear1_relu,linear2_weight.T) + linear2_bias
    # 第三次加
    third_Add = norm2 + linear2
    # 第三次layer_norm
    norm3_mean = torch.mean(third_Add,dim=-1,keepdim=True)
    norm3_std = torch.sqrt(torch.var(third_Add,unbiased=False,dim=-1,keepdim=True) + 1e-5)
    norm3_weight = decoder_layer.state_dict()[f'decoder.layers.{index}.norm3.weight']
    norm3_bias = decoder_layer.state_dict()[f'decoder.layers.{index}.norm3.bias']
    norm3 = ((third_Add - norm3_mean)/norm3_std) * norm3_weight + norm3_bias
    return norm3

decoder_out = torch.empty_like(decoder_input)
decoder_out[:] = decoder_input
for i in range(layer_num):
    decoder_out = my_decoder_layer(transformer_model,decoder_out,encoder_out,i)
decoder_mean = torch.mean(decoder_out,dim=-1,keepdim=True)
decoder_std = torch.sqrt(torch.var(decoder_out,unbiased=False,dim=-1,keepdim=True) + 1e-5)
decoder_weight = transformer_model.state_dict()['decoder.norm.weight']
decoder_bias = transformer_model.state_dict()['decoder.norm.bias']
decoder_out = ((decoder_out - decoder_mean)/decoder_std) * decoder_weight + decoder_bias
print(decoder_out)