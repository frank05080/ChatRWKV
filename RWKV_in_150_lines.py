########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# 导入库
import numpy as np
# 这行代码设置了numpy数组的打印格式，其中precision=4表示小数点后保留4位，
# suppress=True表示抑制小数点的科学计数法表示，linewidth=200表示每行的字符宽度为200。
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer

# 加载一个分词器
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# 使用types.SimpleNamespace()创建一个简单的命名空间对象args，并为其设置以下属性：
args = types.SimpleNamespace()
# 模型的路径。
args.MODEL_NAME = './HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
args.n_layer = 24 # 模型的层数。
args.n_embd = 1024 # 模型的嵌入维度。

# 定义了需要续写的字符串，描述了科学家在西藏的一个偏远山谷中发现了一群会说流利中文的龙的情况。
context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
NUM_TRIALS = 3 # 尝试生成文本的次数。
LENGTH_PER_TRIAL = 100 # 每次尝试生成的文本长度。
TEMPERATURE = 1.0 # 控制生成文本的随机性的参数。值越大，生成的文本越随机；值越小，生成的文本越确定。
TOP_P = 0.85 # 在生成文本时，只考虑累积概率超过此值的词汇。

########################################################################################################

class RWKV_RNN(torch.jit.ScriptModule):
    def __init__(self, args):
        super().__init__()
        # 将传入的args参数赋值给类的属性args。
        self.args = args
        # 将模型设置为评估模式，这意味着模型中的dropout和batchnorm将被禁用。
        self.eval() # set torch to inference mode
        
        # 从指定路径加载模型权重，并确保权重被加载到CPU上。
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        # 这几行代码对加载的权重进行了处理。它们检查权重的键名，并根据键名对权重进行不同的操作。
        for k in w.keys():
            if      '.time_' in k: # like 'blocks.0.att.time_decay'
                w[k] = w[k].squeeze() # like in 'blocks.0.att.time_mix_k', [1,1,1024] -> [1024]
            if '.time_decay' in k: # sometimes it satisfies both this and the above
                w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
            else: 
                w[k] = w[k].float() # convert to f32 type (before it was bfloat16)
        
        # 创建一个新的命名空间对象，并将其赋值给self.w。
        self.w = types.SimpleNamespace() # set self.w from w
        # 在self.w中创建一个名为blocks的字典。
        self.w.blocks = {}
        # for k in w.keys(): - 遍历字典w的所有键。注释中的例子 
        # "blocks.0.att.time_first" => self.w.blocks[0].att.time_first" 
        # 说明了代码的目标：将点分隔的键 转换为 嵌套的属性访问。
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.') #  使用.作为分隔符将键k分割成多个部分，并将结果存储在parts列表中。
            last = parts.pop() # 从parts列表中弹出最后一个元素并存储在last中。这将是要设置的属性的名称。
            #  初始化一个变量here，它将用于遍历或创建self.w中的嵌套命名空间。
            here = self.w
            # 遍历parts列表中的每个部分。
            for p in parts:
                # 检查当前部分p是否是数字。
                if p.isdigit():
                    p = int(p)
                    # 如果当前数字键p不在here中，则在here中为其创建一个新的命名空间。
                    if p not in here: here[p] = types.SimpleNamespace()
                    # 更新here以指向新创建的或已存在的命名空间。
                    here = here[p]
                # 如果当前部分p不是数字。
                else:
                    # 如果here (here at start is w namespace) 没有名为p的属性，则为其创建一个新的命名空间。
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias) # self.args.n_embd = 1024

    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
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

    # 定义forward方法，它接受两个参数：token和state。
    def forward(self, token, state):
        # 这是一个上下文管理器，确保在此代码块中不会计算任何梯度。
        # 这通常用于评估模式，以提高性能并避免不必要的计算。
        with torch.no_grad():
            # 如果state为None，则初始化state为一个全零张量。
            # 其形状由 self.args.n_layer 和 self.args.n_embd 确定。
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd) # torch.float32, shape: [120, 1024]
                # 遍历每一层，并将state的特定位置设置为-1e30（表示负无穷大）。
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # -infinity
            # 使用token索引self.w.emb.weight(has shape [50277, 1024])，获取词嵌入向量。
            x = self.w.emb.weight[token] # has shape [1024]
            # 对获取的词嵌入向量x应用层归一化。
            x = self.layer_norm(x, self.w.blocks[0].ln0) # second param is a simple namespace
            
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att # 获取当前层的注意力参数, type of simple namespace
                # 这些行使用time_mixing方法对x进行处理，并将结果加到x上。
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                ffn = self.w.blocks[i].ffn # 获取当前层的前馈网络参数。type of simpleNamespace
                # 使用channel_mixing方法对x进行处理，并将结果加到x上。
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            # 对x应用最后的层归一化，并与self.w.head.weight进行矩阵乘法。
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state # x's shape change from [1024] (feature size) to [50277] (logits) # state's shape dont change, still [120,1024]
##########################################################################################################

# 这段代码是一个用于生成随机样本的函数。
# 这是一个函数定义，函数名为 sample_logits，接受三个参数 out、temperature 
# 和 top_p，其中 temperature 默认值为 1.0，top_p 默认值为 0.8。
def sample_logits(out, temperature=1.0, top_p=0.8): # out's shape: [50277]
    # 这行代码使用 softmax 函数对 out 进行操作，将输出转换为概率分布。
    # dim=-1 表示在最后一个维度上进行 softmax 操作。.numpy() 将结果转换为 NumPy 数组。
    probs = F.softmax(out, dim=-1).numpy() # probs' shape: (50277,) that is numpy arr
    
    # 这行代码使用 NumPy 的 np.sort 函数对概率分布进行排序，
    # 并通过 [::-1] 实现降序排列。结果保存在 sorted_probs 变量中。
    sorted_probs = np.sort(probs)[::-1] # array([0.9578, 0.0082, 0.0054, ..., 0.    , 0.    , 0.    ], dtype=float32)
    # 这行代码计算累积概率，使用 NumPy 的 np.cumsum 函数对 sorted_probs 
    # 进行累加操作。结果保存在 cumulative_probs 变量中。
    cumulative_probs = np.cumsum(sorted_probs) # array([0.9578, 0.966 , 0.9714, ..., 1.    , 1.    , 1.    ], dtype=float32)
    # 这行代码通过比较 cumulative_probs 是否大于 top_p 来找到概率分布中的截断点。
    # np.argmax 返回第一个满足条件的索引，float() 将其转换为浮点数并保存在 cutoff 变量中。
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    # 这行代码将低于 cutoff 的概率值设为 0，即将概率分布中小于截断点的概率置零。
    probs[probs < cutoff] = 0
    # 这段代码根据 temperature 的取值对概率分布进行调整。
    # 如果 temperature 不等于 1.0，则将概率分布的每个元素取倒数的 1.0 / temperature 次幂。
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    # 这行代码将概率分布归一化，确保所有概率的总和为 1。
    probs = probs / np.sum(probs)
    # 这行代码使用 np.random.choice 函数根据概率分布 probs 生成一个随机样本，
    # a=len(probs) 表示可选的样本范围为 probs 的长度，p=probs 表示每个样本被选中的概率。
    out = np.random.choice(a=len(probs), p=probs)
    # 函数返回生成的随机样本。
    return out

########################################################################################################

# 打印使用 CPU 加载模型的信息，其中 args.MODEL_NAME 是模型名称。
print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
# 创建一个名为 model 的 RWKV_RNN 模型实例，参数为 args。
model = RWKV_RNN(args)

# 打印预处理上下文信息的提示，提示使用的是较慢的版本。然后初始化 init_state 为 None。
print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = None
# 对上下文进行分词编码，并使用模型的 forward 方法逐个处理分词编码的 tokens，
# 将结果保存在 init_out 和 init_state 中。The later inference will use this init_state
for token in tokenizer.encode(context).ids: # token is a int # warm-up process
    init_out, init_state = model.forward(token, init_state)

# 使用循环进行多次试验（NUM_TRIALS 次）。
for TRIAL in range(NUM_TRIALS):
    # 在每次试验的开始打印试验信息和上下文。创建一个空列表 all_tokens 用于保存生成的 tokens。
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    # 初始化变量 out_last 为 0，out 和 state 分别为 init_out 和 init_state 的克隆。
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    # 在每个试验中，使用循环生成 LENGTH_PER_TRIAL 个 tokens。
    for i in range(LENGTH_PER_TRIAL):
        # 调用 sample_logits 函数生成一个随机 token，并将其添加到 all_tokens 列表中。
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        # 使用 tokenizer.decode 将 all_tokens[out_last:] 解码为文本，
        # 并检查解码结果是否包含无效的 utf-8 字符（'\ufffd'）。如果结果有效，则将其打印出来。
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        # 调用模型的 forward 方法，将生成的 token 和当前的状态传递给模型，获取更新的 out 和 state。
        out, state = model.forward(token, state)       
print('\n')