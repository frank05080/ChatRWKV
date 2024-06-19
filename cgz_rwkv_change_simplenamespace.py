import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

EXPORT_ONNX = False
EXPORT_TS = False
ONNX_NAME = "rwkv_model.onnx"

tokenizer = Tokenizer.from_file("20B_tokenizer.json")

n_embd = 1024
N_LAYER = 24

args = {
    'MODEL_NAME': './HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066',
    'n_layer': N_LAYER, # can't set too big, baoneicun!!
    'n_embd': n_embd
}

context = "Horizon Robotics is the world's largest manufacturer of spinning"
NUM_TRIALS = 1
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.85

########################################################################################################

class AttentionBlock(torch.nn.Module):
    def __init__(self, w, layer_idx):
        super().__init__()
        self.ln1_weight = w.get(f'blocks.{layer_idx}.ln1.weight')
        self.ln1_bias = w.get(f'blocks.{layer_idx}.ln1.bias')
        self.time_mix_k = w.get(f'blocks.{layer_idx}.att.time_mix_k')
        self.time_mix_v = w.get(f'blocks.{layer_idx}.att.time_mix_v')
        self.time_mix_r = w.get(f'blocks.{layer_idx}.att.time_mix_r')
        self.time_first = w.get(f'blocks.{layer_idx}.att.time_first')
        self.time_decay = w.get(f'blocks.{layer_idx}.att.time_decay')
        self.key_weight = w.get(f'blocks.{layer_idx}.att.key.weight')
        self.value_weight = w.get(f'blocks.{layer_idx}.att.value.weight')
        self.receptance_weight = w.get(f'blocks.{layer_idx}.att.receptance.weight')
        self.output_weight = w.get(f'blocks.{layer_idx}.att.output.weight')

class FeedForwardBlock(torch.nn.Module):
    def __init__(self, w, layer_idx):
        super().__init__()
        self.ln2_weight = w.get(f'blocks.{layer_idx}.ln2.weight')
        self.ln2_bias = w.get(f'blocks.{layer_idx}.ln2.bias')
        self.time_mix_k = w.get(f'blocks.{layer_idx}.ffn.time_mix_k')
        self.time_mix_r = w.get(f'blocks.{layer_idx}.ffn.time_mix_r')
        self.key_weight = w.get(f'blocks.{layer_idx}.ffn.key.weight')
        self.value_weight = w.get(f'blocks.{layer_idx}.ffn.value.weight')
        self.receptance_weight = w.get(f'blocks.{layer_idx}.ffn.receptance.weight')

class RWKVBlock(torch.nn.Module):
    def __init__(self, w, layer_idx):
        super().__init__()
        self.ln0_weight = w.get(f'blocks.{layer_idx}.ln0.weight')
        self.ln0_bias = w.get(f'blocks.{layer_idx}.ln0.bias')
        self.att = AttentionBlock(w, layer_idx)
        self.ffn = FeedForwardBlock(w, layer_idx)

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_layer = args['n_layer']
        self.n_embd = args['n_embd']
        self.eval()  # set torch to inference mode
        
        w = torch.load(args['MODEL_NAME'] + '.pth', map_location='cpu')
        for k in w.keys():
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float())  # the real time decay is like e^{-e^x}
            else: w[k] = w[k].float()  # convert to f32 type
        
        print("Keys in the weights dictionary:")
        print(list(w.keys()))  # Print the keys to verify

        self.emb = torch.nn.Embedding.from_pretrained(w['emb.weight'])
        self.ln_out_weight = w['ln_out.weight']
        self.ln_out_bias = w['ln_out.bias']
        self.head_weight = w['head.weight']
        
        self.blocks = torch.nn.ModuleList([RWKVBlock(w, i) for i in range(self.n_layer)])

    def layer_norm(self, x, weight, bias):
        return F.layer_norm(x, (self.n_embd,), weight=weight, bias=bias)

    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv
        
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq
        return ow @ (r * wkv)

    def forward(self, token, state):
        # if state is None:
        #     state = torch.zeros(self.n_layer * 5, self.n_embd)
        #     for i in range(self.n_layer): state[5*i+4] = -1e30  # -infinity
        
        x = self.emb(token)
        x = self.layer_norm(x, self.blocks[0].ln0_weight, self.blocks[0].ln0_bias)
        for index, block in enumerate(self.blocks):
            x = x + self.time_mixing(self.layer_norm(x, block.att.ln1_weight, block.att.ln1_bias), state, index, 
                block.att.time_mix_k, block.att.time_mix_v, block.att.time_mix_r, block.att.time_first, block.att.time_decay, 
                block.att.key_weight, block.att.value_weight, block.att.receptance_weight, block.att.output_weight)
            x = x + self.channel_mixing(self.layer_norm(x, block.ffn.ln2_weight, block.ffn.ln2_bias), state, index, 
                block.ffn.time_mix_k, block.ffn.time_mix_r, 
                block.ffn.key_weight, block.ffn.value_weight, block.ffn.receptance_weight)
        
        x = self.head_weight @ self.layer_norm(x, self.ln_out_weight, self.ln_out_bias)
        return x.float(), state

####################################################################################3
# Module 'RWKV_RNN' has no attribute 'w' (This attribute exists on the Python module, but we failed to convert Python type: 'dict' to a TorchScript type. Dictionary inputs to traced functions must have consistent type. Found Tensor and Dict[str, Tensor]. Its type was inferred; try adding a type annotation for the attribute.):

# import numpy as np
# np.set_printoptions(precision=4, suppress=True, linewidth=200)
# import torch
# from torch.nn import functional as F
# from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# args = {
#     'MODEL_NAME': './HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066',
#     'n_layer': 24,
#     'n_embd': 1024
# }

# context = "\nHorizon Robotics is"
# NUM_TRIALS = 3
# LENGTH_PER_TRIAL = 100
# TEMPERATURE = 1.0
# TOP_P = 0.85


# class RWKV_RNN(torch.nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_layer = args['n_layer']
#         self.n_embd = args['n_embd']
#         self.eval()  # set torch to inference mode
        
#         w = torch.load(args['MODEL_NAME'] + '.pth', map_location='cpu')
#         for k in w.keys():
#             if '.time_' in k: w[k] = w[k].squeeze()
#             if '.time_decay' in k: w[k] = -torch.exp(w[k].float())  # the real time decay is like e^{-e^x}
#             else: w[k] = w[k].float()  # convert to f32 type
        
#         self.w = {'blocks': {}, 'emb': {}, 'head': {}, 'ln_out': {}}
#         for k in w.keys():  # example: "blocks.0.att.time_first" => self.w['blocks'][0]['att']['time_first']
#             parts = k.split('.')
#             last = parts.pop()
#             here = self.w
#             for p in parts:
#                 if p.isdigit():
#                     p = int(p)
#                     if p not in here: here[p] = {}
#                     here = here[p]
#                 else:
#                     if p not in here: here[p] = {}
#                     here = here[p]
#             here[last] = w[k]

#     def layer_norm(self, x, w):
#         return F.layer_norm(x, (self.n_embd,), weight=w['weight'], bias=w['bias'])

#     def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
#         xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
#         xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
#         state[5*i+0] = x
#         r = torch.sigmoid(rw @ xr)
#         k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
#         return r * (vw @ k)

#     def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
#         xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
#         xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
#         xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
#         state[5*i+1] = x
#         r = torch.sigmoid(rw @ xr)
#         k = kw @ xk
#         v = vw @ xv
        
#         aa = state[5*i+2]
#         bb = state[5*i+3]
#         pp = state[5*i+4]
#         ww = time_first + k
#         qq = torch.maximum(pp, ww)
#         e1 = torch.exp(pp - qq)
#         e2 = torch.exp(ww - qq)
#         a = e1 * aa + e2 * v
#         b = e1 * bb + e2
#         wkv = a / b
#         ww = pp + time_decay
#         qq = torch.maximum(ww, k)
#         e1 = torch.exp(ww - qq)
#         e2 = torch.exp(k - qq)
#         state[5*i+2] = e1 * aa + e2 * v
#         state[5*i+3] = e1 * bb + e2
#         state[5*i+4] = qq
#         return ow @ (r * wkv)

#     def forward(self, token, state):
#         if state is None:
#             state = torch.zeros(self.n_layer * 5, self.n_embd)
#             for i in range(self.n_layer): state[5*i+4] = -1e30  # -infinity
        
#         x = self.w['emb']['weight'][token]
#         x = self.layer_norm(x, self.w['blocks'][0]['ln0'])
#         for i in range(self.n_layer):
#             att = self.w['blocks'][i]['att']
#             x = x + self.time_mixing(self.layer_norm(x, self.w['blocks'][i]['ln1']), state, i, 
#                 att['time_mix_k'], att['time_mix_v'], att['time_mix_r'], att['time_first'], att['time_decay'], 
#                 att['key']['weight'], att['value']['weight'], att['receptance']['weight'], att['output']['weight'])
#             ffn = self.w['blocks'][i]['ffn']
#             x = x + self.channel_mixing(self.layer_norm(x, self.w['blocks'][i]['ln2']), state, i, 
#                 ffn['time_mix_k'], ffn['time_mix_r'], 
#                 ffn['key']['weight'], ffn['value']['weight'], ffn['receptance']['weight'])
        
#         x = self.w['head']['weight'] @ self.layer_norm(x, self.w['ln_out'])
#         return x.float(), state

###################################################################################

## input not work for int

# import numpy as np
# np.set_printoptions(precision=4, suppress=True, linewidth=200)
# import torch
# from torch.nn import functional as F
# from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# args = {
#     'MODEL_NAME': './HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066',
#     'n_layer': 24,
#     'n_embd': 1024
# }

# context = "\nHorizon Robotics is"
# NUM_TRIALS = 3
# LENGTH_PER_TRIAL = 100
# TEMPERATURE = 1.0
# TOP_P = 0.85


# class RWKV_RNN(torch.nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_layer = args['n_layer']
#         self.n_embd = args['n_embd']
#         self.eval()  # set torch to inference mode
        
#         w = torch.load(args['MODEL_NAME'] + '.pth', map_location='cpu')
#         for k in w.keys():
#             if '.time_' in k: w[k] = w[k].squeeze()
#             if '.time_decay' in k: w[k] = -torch.exp(w[k].float())  # the real time decay is like e^{-e^x}
#             else: w[k] = w[k].float()  # convert to f32 type
        
#         self.w = {'blocks': {}}
#         for k in w.keys():  # example: "blocks.0.att.time_first" => self.w['blocks'][0]['att']['time_first']
#             parts = k.split('.')
#             last = parts.pop()
#             here = self.w
#             for p in parts:
#                 if p.isdigit():
#                     p = int(p)
#                     if p not in here: here[p] = {}
#                     here = here[p]
#                 else:
#                     if p not in here: here[p] = {}
#                     here = here[p]
#             here[last] = w[k]

#     def layer_norm(self, x, w):
#         return F.layer_norm(x, (self.n_embd,), weight=w['weight'], bias=w['bias'])

#     def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
#         xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
#         xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
#         state[5*i+0] = x
#         r = torch.sigmoid(rw @ xr)
#         k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
#         return r * (vw @ k)

#     def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
#         xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
#         xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
#         xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
#         state[5*i+1] = x
#         r = torch.sigmoid(rw @ xr)
#         k = kw @ xk
#         v = vw @ xv
        
#         aa = state[5*i+2]
#         bb = state[5*i+3]
#         pp = state[5*i+4]
#         ww = time_first + k
#         qq = torch.maximum(pp, ww)
#         e1 = torch.exp(pp - qq)
#         e2 = torch.exp(ww - qq)
#         a = e1 * aa + e2 * v
#         b = e1 * bb + e2
#         wkv = a / b
#         ww = pp + time_decay
#         qq = torch.maximum(ww, k)
#         e1 = torch.exp(ww - qq)
#         e2 = torch.exp(k - qq)
#         state[5*i+2] = e1 * aa + e2 * v
#         state[5*i+3] = e1 * bb + e2
#         state[5*i+4] = qq
#         return ow @ (r * wkv)

#     def forward(self, token, state):
#         if state is None:
#             state = torch.zeros(self.n_layer * 5, self.n_embd)
#             for i in range(self.n_layer): state[5*i+4] = -1e30  # -infinity
        
#         x = self.w['emb']['weight'][token]
#         x = self.layer_norm(x, self.w['blocks'][0]['ln0'])
#         for i in range(self.n_layer):
#             att = self.w['blocks'][i]['att']
#             x = x + self.time_mixing(self.layer_norm(x, self.w['blocks'][i]['ln1']), state, i, 
#                 att['time_mix_k'], att['time_mix_v'], att['time_mix_r'], att['time_first'], att['time_decay'], 
#                 att['key']['weight'], att['value']['weight'], att['receptance']['weight'], att['output']['weight'])
#             ffn = self.w['blocks'][i]['ffn']
#             x = x + self.channel_mixing(self.layer_norm(x, self.w['blocks'][i]['ln2']), state, i, 
#                 ffn['time_mix_k'], ffn['time_mix_r'], 
#                 ffn['key']['weight'], ffn['value']['weight'], ffn['receptance']['weight'])
        
#         x = self.w['head']['weight'] @ self.layer_norm(x, self.w['ln_out'])
#         return x.float(), state


##########################################################################################################

print(f'\nUsing CPU. Loading {args["MODEL_NAME"]} ...')
model = RWKV_RNN(args)

###########################################################################################################


################### Torch Test Code ###########################

def sample_logits1(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = torch.zeros(120, n_embd, dtype=torch.float32)
for i in range(N_LAYER): init_state[5*i+4] = -1e30

for token in tokenizer.encode(context).ids:
    token = torch.tensor(token) # with [], shape is torch.Size([1]), without, shape is torch.Size([])
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits1(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(torch.tensor(token), state)       
print('\n')

###########################################################################################################

print("get inputs")


if EXPORT_TS:
    # Trace the model -- cgz notes: actually no need to trace
    # print("ready to trace")
    traced_model = torch.jit.trace(model, (torch.tensor(token), state))
    # scripted_model = torch.jit.script(model)

    # 使用 torch.save 保存 trace 模型
    torch.jit.save(traced_model, 'traced_model.ts')
else:
    traced_model = torch.jit.load("traced_model.ts")



init_state = torch.zeros(120, n_embd, dtype=torch.float32)
for i in range(N_LAYER): init_state[5*i+4] = -1e30

for token in tokenizer.encode(context).ids:
    token = torch.tensor(token) # with [], shape is torch.Size([1]), without, shape is torch.Size([])
    init_out, init_state = traced_model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial1111 {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits1(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = traced_model.forward(torch.tensor(token), state)       
print('\n')

################### Onnx Test Code ###########################

if EXPORT_ONNX:
    print("about to export onnx")
    # Export the traced model to ONNX
    torch.onnx.export(
        traced_model,
        (torch.tensor(token), state),  # Tuple of inputs
        ONNX_NAME,
        verbose=False,
        export_params=True,
        opset_version=11, # opset 10 error
        # do_constant_folding=True, # cant do constant folding
        input_names=['int_input', 'tensor_input'],
        output_names=['output']
    )
    print("finish export onnx")

# torch.onnx.export(
#     scripted_model,
#     (dummy_int_input, dummy_tensor_input),  # Tuple of inputs
#     ONNX_NAME,
#     input_names=['int_input', 'tensor_input'],
#     output_names=['output']
# )

import onnx

print("Ready to load")
onnx_model = onnx.load(ONNX_NAME)
print("end load")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession(ONNX_NAME, providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if len(tensor.shape) == 0:
        return tensor.cpu().numpy()
    return tensor.detach().cpu().numpy() # if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(token)), ort_session.get_inputs()[1].name: to_numpy(state)} # from torch tensor to numpy array
ort_outs = ort_session.run(None, ort_inputs)

print("we are here")
# print(type(ort_outs))
# print(ort_outs[0].shape)

out, new_state = model.forward(torch.tensor(token), state)    


print("out: ", to_numpy(out))
print("ort_outs[0]: ", ort_outs[0])
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-01, atol=1e-01) #match
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
# print()
# np.testing.assert_allclose(1, ort_outs[0], rtol=1e-03, atol=1e-05) # not match



# def sample_logits(out, temperature=1.0, top_p=0.8):
#     probs = F.softmax(out, dim=-1).numpy()
#     sorted_probs = np.sort(probs)[::-1]
#     cumulative_probs = np.cumsum(sorted_probs)
#     cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#     probs[probs < cutoff] = 0
#     if temperature != 1.0:
#         probs = probs.pow(1.0 / temperature)
#     probs = probs / np.sum(probs)
#     out = np.random.choice(a=len(probs), p=probs)
#     return out


print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
state = torch.zeros(120, n_embd, dtype=torch.float32)
for i in range(N_LAYER): state[5*i+4] = -1e30

# for token in tokenizer.encode(context).ids:
#     # token = torch.tensor(token) # with [], shape is torch.Size([1]), without, shape is torch.Size([])
#     # print("token type:", type(token))
#     # print("init_state type:", type(init_state))
#     ## ort_inputs = {ort_session.get_inputs()[0].name: np.array(token), ort_session.get_inputs()[1].name: to_numpy(init_state)} # from torch tensor to numpy array
#     ## ort_outs = ort_session.run(None, ort_inputs)
    
#     out, new_state = model.forward(torch.tensor(token), init_state) 
#     # print(ort_outs[0])
#     # if np.isnan(ort_outs.any()):
#     #     continue
#     # np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-01, atol=1e-01) #match
    
#     init_state = ort_outs[1]
#     init_state = torch.from_numpy(init_state)
#     print("init_state.shape: ", init_state.shape)

for token in tokenizer.encode(context).ids:
    print("token is:", token)
    
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(token)), ort_session.get_inputs()[1].name: to_numpy(state)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # init_out, state = ort_outs
    
    init_out, state = model.forward(torch.tensor(token), state) # warm-up
    print(state)
    print(init_out.shape) # (50277,)
    
print(init_out)

def contains_nan(tensor):
    return torch.isnan(tensor).any()

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state_np = init_out, to_numpy(state)
    print("out contains nan:", contains_nan(out))
    for i in range(LENGTH_PER_TRIAL):
        # print()
        # print("  out is:", out)
        token = sample_logits1(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
            
        state_np = to_numpy(state)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(token)), ort_session.get_inputs()[1].name: state_np} # from torch tensor to numpy array
        ort_outs = ort_session.run(None, ort_inputs)
        out, _ = ort_outs
        # print("orig out:", out)
        out = torch.from_numpy(out)
        
        out_torch, state = model.forward(torch.tensor(token), state)
        
        print("out: ", out)
        print("orout_torch: ", out_torch)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(out_torch), ort_outs[0], rtol=1e-01, atol=1e-01) #match
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        
print('\n')

