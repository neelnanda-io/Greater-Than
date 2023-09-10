# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
model.to_str_tokens("The war was fought from the year 1617 to 16")
# %%
number_tokens = torch.tensor([model.to_single_token(str(100+x)[1:]) for x in range(1, 100)]).cuda()
line(number_tokens)

prompts = [f"The war was fought from the year {x} to 16" for x in range(1601, 1700)]
tokens = model.to_tokens(prompts)
logits, cache = model.run_with_cache(tokens)
log_probs = logits[:, -1, :].log_softmax(dim=-1)
imshow(-log_probs[:, number_tokens])
number_log_probs = -log_probs[:, number_tokens]
RANGE = np.arange(1, 100)
line([number_log_probs.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)])
# %%
unembed_dirs = model.W_U[:, number_tokens[2:]] - model.W_U[:, number_tokens[:-2]]
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
resid_stack = resid_stack[:, 1:-1]
dla = (resid_stack * unembed_dirs.T).sum(-1).mean(-1)
line(dla, x=resid_labels, title="DLA of +1 rel -1")
temp_df = pd.DataFrame({'dla': to_numpy(dla), 'resid_labels': resid_labels})
nutils.show_df(temp_df.sort_values("dla", ascending=False).head(20))
# %%
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=True, apply_ln=True, pos_slice=-1, return_labels=True)
resid_stack = resid_stack[:, 1:-1]
dla = (resid_stack * unembed_dirs.T).sum(-1).mean(-1)
line(dla, x=resid_labels, title="DLA of +1 rel -1")
temp_df = pd.DataFrame({'dla': to_numpy(dla), 'resid_labels': resid_labels})
nutils.show_df(temp_df.sort_values("dla", ascending=False).head(20))

# %%
layer = 9
head = 1
z = cache["z", layer][:, -1, head, :]
head_dla = z @ model.W_O[layer, head] @ model.W_U[:, number_tokens] / cache["scale"][:, -1, :]
imshow(head_dla, x=RANGE, y=RANGE, title=f"Head DLA for L{layer}H{head}")
line([head_dla.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)], title=f"Head DLA Diags for L{layer}H{head}")
imshow(cache["pattern", layer][:, head, -1, :], y=RANGE, x=nutils.process_tokens_index(prompts[0]), title=f"Head Pattern for L{layer}H{head}")

# %%
n=100
fourier_basis = []
fourier_labels = []
fourier_basis.append(torch.ones(n))
fourier_labels.append("const")
for i in range(1, n//2):
    fourier_basis.append(torch.cos(torch.arange(n) * i * 2 * np.pi / n))
    fourier_basis.append(torch.sin(torch.arange(n) * i * 2 * np.pi / n))
    fourier_labels.append(f"cos({i})")
    fourier_labels.append(f"sin({i})")
fourier_basis.append(torch.tensor([-1 if i%2 else 1 for i in range(n)]))
fourier_labels.append("+-1")
fourier_basis = torch.stack(fourier_basis).cuda()
fourier_basis /= torch.norm(fourier_basis, dim=-1, keepdim=True)
imshow(fourier_basis @ fourier_basis.T)
n=10
fourier_basis_10 = []
fourier_labels_10 = []
fourier_basis_10.append(torch.ones(n))
fourier_labels_10.append("const")
for i in range(1, n//2):
    fourier_basis_10.append(torch.cos(torch.arange(n) * i * 2 * np.pi / n))
    fourier_basis_10.append(torch.sin(torch.arange(n) * i * 2 * np.pi / n))
    fourier_labels_10.append(f"cos({i})")
    fourier_labels_10.append(f"sin({i})")
fourier_basis_10.append(torch.tensor([-1 if i%2 else 1 for i in range(n)]))
fourier_labels_10.append("+-1")
fourier_basis_10 = torch.stack(fourier_basis_10).cuda()
fourier_basis_10 /= torch.norm(fourier_basis_10, dim=-1, keepdim=True)
imshow(fourier_basis_10 @ fourier_basis_10.T)
# %%
def pad_to_1d(tensor, n=100):
    # assert len(tensor.shape)==1
    # assert len(tensor)<=n
    padded_tensor = torch.zeros(tensor.shape[:-1]+(n,)).to(tensor.device)
    padded_tensor[..., -tensor.shape[-1]:] = tensor
    return padded_tensor
def pad_to_2d(tensor, n=100):
    # assert tensor.shape[0]<=n
    # assert tensor.shape[1]<=n
    padded_tensor = torch.zeros(tensor.shape[:-2]+(n,n)).to(tensor.device)
    padded_tensor[..., -tensor.shape[0]:, -tensor.shape[1]:] = tensor
    return padded_tensor
line(pad_to_1d(torch.arange(10)))
imshow(pad_to_2d(torch.arange(10)[None, :], 15))

# %%
# %%
layer = 9
ni = 860
acts = cache["post", layer][:, -1, ni]
neuron_dla = acts[:, None] @ model.W_out[layer, ni][None, :] @ model.W_U[:, number_tokens] / cache["scale"][:, -1, :]
imshow(neuron_dla, x=RANGE, y=RANGE, title=f"Neuron DLA for L{layer}N{ni}")
line([neuron_dla.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)], title=f"Neuron DLA Diags for L{layer}N{ni}")
line(acts, x=RANGE)
line(pad_to_1d(acts) @ fourier_basis.T, x=fourier_labels, title=f"Fourier Basis Neuron Acts for L{layer}N{ni}")
line(pad_to_1d(acts).reshape(10, 10), title=f"Fourier Basis Neuron Acts for L{layer}N{ni}")
# %%
wout = model.W_out[layer, ni]
neuron_wdla = wout @ model.W_U
line(pad_to_1d(neuron_wdla[number_tokens]).reshape(10, 10), title="Neuron outputs")
nutils.show_df(nutils.create_vocab_df(neuron_wdla).head(100))
# %%
days = [' Monday', " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
day_tokens = model.to_tokens(days, prepend_bos=False).squeeze(0)
line(neuron_wdla[day_tokens], x=days, title=f"Neuron L{layer}N{ni} wDLA for days")

months = [" January", " February", " March", " April", " May", " June", " July", " August", " September", " October", " November", " December"]
month_tokens = model.to_tokens(months, prepend_bos=False).squeeze(0)
line(neuron_wdla[month_tokens], x=months, title=f"Neuron L{layer}N{ni} wDLA for months")
# %%
win = model.W_in[layer, :, ni]
neuron_resid_stack, neuron_resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, layer=layer, mlp_input=True, apply_ln=True, pos_slice=-1, return_labels=True)
line((neuron_resid_stack @ win)[:, 2::10].mean(-1) - (neuron_resid_stack @ win)[:, 0::10].mean(-1), x=neuron_resid_labels)
l8_neuron_acts = cache["post", 8][:, -1, :][2::10].mean(0) - cache["post", 8][:, -1, :][0::10].mean(0)
l8_wdna = model.W_out[8] @ win
line(l8_neuron_acts * l8_wdna / cache["scale", 9, "ln2"][2::10, -1, 0].mean())
# %%
layer2 = 8
ni2 = 1676
line(cache["post", layer2][:, -1, ni2])
line(pad_to_1d(cache["post", layer2][:, -1, ni2]) @ fourier_basis.T, x=fourier_labels)
nutils.create_vocab_df(model.W_out[layer2, ni2] @ model.W_U).head(100)
line(model.W_out[layer2, ni2] @ model.W_U[:, number_tokens], x=RANGE)
# %%
for layer in range(8, 12):
    mlp_out  = cache["mlp_out", layer][:, -1, :]
    mlp_dla = mlp_out @ model.W_U[:, number_tokens] / cache["scale"][:, -1, :]
    mlp_dla -= mlp_dla.min(dim=-1, keepdim=True)[0]
    imshow(mlp_dla, x=RANGE, y=RANGE, title=f"MLP DLA for L{layer}")
    line([mlp_dla.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)], title=f"MLP DLA Diags for L{layer}")
    imshow(fourier_basis @ pad_to_2d(mlp_dla) @ fourier_basis.T, x=fourier_labels, y=fourier_labels)

# %%
resids = cache["resid_pre", 8][:, -1, :]
resids_cent = resids - resids.mean(0)
U, S, Vh = torch.linalg.svd(resids_cent)
line(S)
line(pad_to_1d(U[:, :20].T) @ fourier_basis.T, x=fourier_labels, title="SVD of Resids")
line((pad_to_1d(resids_cent.T) @ fourier_basis.T).norm(dim=0), x=fourier_labels, title="Fourier Norm of Resids")
# %%
layer = 9 
ni = 1919
win = model.W_in[layer, :, ni]
neuron_resid_stack, neuron_resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, layer=layer, mlp_input=True, apply_ln=True, pos_slice=-1, return_labels=True)
line((neuron_resid_stack @ win)[:, 2::10].mean(-1) - (neuron_resid_stack @ win)[:, 0::10].mean(-1), x=neuron_resid_labels)
l8_neuron_acts = cache["post", 8][:, -1, :][2::10].mean(0) - cache["post", 8][:, -1, :][0::10].mean(0)
l8_wdna = model.W_out[8] @ win
line(l8_neuron_acts * l8_wdna / cache["scale", 9, "ln2"][2::10, -1, 0].mean())
top_neurons = (l8_neuron_acts * l8_wdna).argsort()[-6:]
line(cache["post", 8][:, -1, top_neurons].T, line_labels=["L8N"+str(i.item()) for i in top_neurons])
# %%
acts = cache["post", layer][:, -1, ni]
wdla = model.W_out[layer, ni] @ model.W_U[:, number_tokens]
line([acts, wdla], x=RANGE, line_labels=["acts", "wdla"])
line([acts @ fourier_basis.T[1:] / (acts @ fourier_basis.T[1:]).pow(2).sum().sqrt(), (wdla @ fourier_basis.T[1:])/(wdla @ fourier_basis.T[1:]).pow(2).sum().sqrt()], x=fourier_labels, line_labels=["acts", "wdla"])
# %%
neuron_fourier_wdla = model.W_out @ model.W_U[:, number_tokens] @ fourier_basis.T[1:]
neuron_fourier_wdla.shape
# %%
line((neuron_fourier_wdla[:, :, 39].pow(2)+neuron_fourier_wdla[:, :, 40].pow(2))/neuron_fourier_wdla[:, :, :].pow(2).sum(-1))
line((neuron_fourier_wdla[:, :, 19].pow(2)+neuron_fourier_wdla[:, :, 20].pow(2))/neuron_fourier_wdla[:, :, :].pow(2).sum(-1))
# %%
