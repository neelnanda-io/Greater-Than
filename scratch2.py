

# # %%
# import os
# os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small").to("cpu")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
utils.test_prompt("The war was fought from the year 1617 to 16", "18", model,False)
# %%
number_tokens = torch.tensor([model.to_single_token(str(100+x)[1:]) for x in range(1, 100)])
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
line(cache["post", 8][:, -1, 1676], x=np.arange(1, 100))
# %%
def abl_neuron_hook(mlp_post, hook):
    mlp_post[:, -1, 1676] = 0.
    return mlp_post
model.reset_hooks()
model.blocks[8].mlp.hook_post.add_hook(abl_neuron_hook)

abl_logits, abl_cache = model.run_with_cache(tokens)
abl_log_probs = abl_logits[:, -1, :].log_softmax(dim=-1)
imshow(-abl_log_probs[:, number_tokens])
abl_number_log_probs = -abl_log_probs[:, number_tokens]
RANGE = np.arange(1, 100)
line([abl_number_log_probs.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)])
imshow(abl_number_log_probs)
# %%
prob_diff = []
for i in range(1, 98):
    inc = (-number_log_probs[i, :i]).exp().sum()
    corr = (-number_log_probs[i, i+1:]).exp().sum()
    prob_diff.append(inc)
prob_diff = torch.tensor(prob_diff)
zero_abl_prob_diff = []
for i in range(1, 98):
    inc = (-abl_number_log_probs[i, :i]).exp().sum()
    corr = (-abl_number_log_probs[i, i+1:]).exp().sum()
    zero_abl_prob_diff.append(inc)
zero_abl_prob_diff = torch.tensor(zero_abl_prob_diff)
line([prob_diff, zero_abl_prob_diff], line_labels=["normal", "zero_abl"])
# %%
wout = model.W_out[8, 1676, :]
wout = wout / wout.norm()
W_out = model.W_out
W_out_norm = W_out / W_out.norm(dim=-1, keepdim=True)
line(W_out_norm @ wout)
# %%
resid_post = cache.stack_activation("resid_mid")[:, :, -1, :]
px.box((resid_post @ wout / resid_post.norm(dim=-1)).T)
# %%
W_in = model.W_in[9:11]
W_in = W_in / W_in.norm(dim=1, keepdim=True)
W_in_randn = torch.randn_like(W_in)
W_in_randn = W_in_randn / W_in_randn.norm(dim=1, keepdim=True)
line((wout @ W_in).sort(-1).values)
line((wout @ W_in_randn).sort(-1).values)
# %%
W_in = model.W_in[7:11]
W_in = W_in / W_in.norm(dim=1, keepdim=True)
W_out = model.W_out[7:11]
W_out = W_out / W_out.norm(dim=2, keepdim=True)
win = W_in[1, :, 1676]
wout = W_out[1, 1676, :]
W_in_sim = win @ W_in
W_out_sim = W_out @ wout
scatter(x=W_in_sim, y=W_out_sim, hover=np.arange(d_mlp), facet_col=0, xaxis="W_in sim", yaxis="W_out sim", title="Similarity of neurons to L8N1676", facet_labels=["L7", "L8", "L9", "L10"])
# %%


nutils.show_df(nutils.create_vocab_df(win @ model.W_U).head(20))
nutils.show_df(nutils.create_vocab_df(win @ model.W_U).tail(20))
nutils.show_df(nutils.create_vocab_df(wout @ model.W_U).head(20))
nutils.show_df(nutils.create_vocab_df(wout @ model.W_U).tail(20))
# %%
W_outU = model.W_out[:, :100, :] @ model.W_U
line(W_outU.std(-1))
# %%
nutils.show_df(nutils.create_vocab_df(W_outU[11, 95]).head(100))
nutils.show_df(nutils.create_vocab_df(W_outU[11, 95]).tail(20))

# %%
px.box(W_outU.std(-1).T)
# %%
