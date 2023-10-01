

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

model = HookedTransformer.from_pretrained("gpt2-small").to("cuda")

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

prompts = [f"The war was fought from the year {x} to 16" for x in range(1605, 1695)]
tokens = model.to_tokens(prompts)
base_logits, cache = model.run_with_cache(tokens)
base_log_probs = base_logits[:, -1, :].log_softmax(dim=-1)
imshow(-base_log_probs[:, number_tokens])
number_base_log_probs = -base_log_probs[:, number_tokens]
# RANGE = np.arange(1, 100)
# line([number_base_log_probs.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)])
# %%
# line(cache["post", 8][:, -1, 1676], x=np.arange(1, 100))
# # %%
# def abl_neuron_hook(mlp_post, hook):
#     mlp_post[:, -1, 1676] = 0.
#     return mlp_post
# model.reset_hooks()
# model.blocks[8].mlp.hook_post.add_hook(abl_neuron_hook)

# abl_logits, abl_cache = model.run_with_cache(tokens)
# abl_log_probs = abl_logits[:, -1, :].log_softmax(dim=-1)
# imshow(-abl_log_probs[:, number_tokens])
# abl_number_log_probs = -abl_log_probs[:, number_tokens]
# RANGE = np.arange(1, 100)
# line([abl_number_log_probs.diag(i)[-95:] for i in range(-2, 5)], x=RANGE[-95:], line_labels=[f"diag({i})" for i in range(-2, 5)])
# imshow(abl_number_log_probs)
# # %%
# prob_diff = []
# for i in range(1, 98):
#     inc = (-number_log_probs[i, :i]).exp().sum()
#     corr = (-number_log_probs[i, i+1:]).exp().sum()
#     prob_diff.append(inc)
# prob_diff = torch.tensor(prob_diff)
# zero_abl_prob_diff = []
# for i in range(1, 98):
#     inc = (-abl_number_log_probs[i, :i]).exp().sum()
#     corr = (-abl_number_log_probs[i, i+1:]).exp().sum()
#     zero_abl_prob_diff.append(inc)
# zero_abl_prob_diff = torch.tensor(zero_abl_prob_diff)
# line([prob_diff, zero_abl_prob_diff], line_labels=["normal", "zero_abl"])
# # %%
# wout = model.W_out[8, 1676, :]
# wout = wout / wout.norm()
# W_out = model.W_out
# W_out_norm = W_out / W_out.norm(dim=-1, keepdim=True)
# line(W_out_norm @ wout)
# # %%
# resid_post = cache.stack_activation("resid_mid")[:, :, -1, :]
# px.box((resid_post @ wout / resid_post.norm(dim=-1)).T)
# # %%
# W_in = model.W_in[9:11]
# W_in = W_in / W_in.norm(dim=1, keepdim=True)
# W_in_randn = torch.randn_like(W_in)
# W_in_randn = W_in_randn / W_in_randn.norm(dim=1, keepdim=True)
# line((wout @ W_in).sort(-1).values)
# line((wout @ W_in_randn).sort(-1).values)
# # %%
# W_in = model.W_in[7:11]
# W_in = W_in / W_in.norm(dim=1, keepdim=True)
# W_out = model.W_out[7:11]
# W_out = W_out / W_out.norm(dim=2, keepdim=True)
# win = W_in[1, :, 1676]
# wout = W_out[1, 1676, :]
# W_in_sim = win @ W_in
# W_out_sim = W_out @ wout
# scatter(x=W_in_sim, y=W_out_sim, hover=np.arange(d_mlp), facet_col=0, xaxis="W_in sim", yaxis="W_out sim", title="Similarity of neurons to L8N1676", facet_labels=["L7", "L8", "L9", "L10"])
# # %%


# nutils.show_df(nutils.create_vocab_df(win @ model.W_U).head(20))
# nutils.show_df(nutils.create_vocab_df(win @ model.W_U).tail(20))
# nutils.show_df(nutils.create_vocab_df(wout @ model.W_U).head(20))
# nutils.show_df(nutils.create_vocab_df(wout @ model.W_U).tail(20))
# # %%
# W_outU = model.W_out[:, :100, :] @ model.W_U
# line(W_outU.std(-1))
# # %%
# nutils.show_df(nutils.create_vocab_df(W_outU[11, 95]).head(100))
# nutils.show_df(nutils.create_vocab_df(W_outU[11, 95]).tail(20))

# # %%
# px.box(W_outU.std(-1).T)
# # %%
# def zero_abl_neuron(mlp_post, hook, ni):
#     mlp_post[:, -1, ni] = 0.
#     return mlp_post
# for layer in [9, 10]:
#     for i in tqdm.trange(d_mlp):
#         logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", layer), partial(zero_abl_neuron, ni=i))])
# %%
neuron_df = nutils.make_neuron_df(n_layers, d_mlp).query("L>=9 & L<=10")
neuron_df
# %%
LAYER = 8
NI = 1676
wout = model.W_out[LAYER, NI]
# bout = model.b_out[LAYER, NI]
bin = model.b_in[LAYER, NI]
wout_norm = wout / wout.norm()
win = model.W_in[LAYER, :, NI]
win_norm = win / win.norm()
print([x.item() for x in [win.norm(), bin, wout.norm()]])
fig = px.histogram(to_numpy(model.W_in.norm(dim=1).T), marginal="box", barmode="overlay", title='W_in norm')
fig.add_vline(x=win.norm().item())
fig.show()
fig = px.histogram(to_numpy(model.b_in.T), marginal="box", barmode="overlay", title='b_in')
fig.add_vline(x=bin.item())
fig.show()
fig = px.histogram(to_numpy(model.W_out.norm(dim=-1).T), marginal="box", barmode="overlay", title='W_out norm')
fig.add_vline(x=wout.norm().item())
fig.show()
# %%
data = load_dataset("stas/openwebtext-10k", split="train")
token_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=256).shuffle(42)
dataset_tokens = token_data[:32]["tokens"]
dataset_logits, dataset_cache = model.run_with_cache(dataset_tokens)
dataset_acts = dataset_cache["post", LAYER][:, :, NI]
line(dataset_acts)
token_df = nutils.make_token_df(dataset_tokens)
token_df["act"] = to_numpy(dataset_acts).flatten()
nutils.show_df(token_df.sort_values("act", ascending=False).head(50))
# %%
for i in [9, 17, 31]:
    nutils.create_html(model.to_str_tokens(dataset_tokens[i]), dataset_acts[i])
# %%
local_logits, local_cache = model.run_with_cache(dataset_tokens[9])
local_cache: ActivationCache
resid_stack, resid_labels = local_cache.get_full_resid_decomposition(layer=8, mlp_input=True, apply_ln=True, return_labels=True, expand_neurons=False)
dna_stack_9 = resid_stack[:, 0] @ win
acts_9 = dataset_acts[9]
top_acts = acts_9>3
line([dna_stack_9[:, top_acts].mean(1), dna_stack_9[:, ~top_acts].mean(1)], line_labels=["top", "not top"], title="DNA for dataset example 9 and neuron L8N1676", x=resid_labels)
component_df = pd.DataFrame({"label":resid_labels, "dna":to_numpy(dna_stack_9[:, top_acts].mean(1)), "dna_not_top":to_numpy(dna_stack_9[:, ~top_acts].mean(1))})
nutils.show_df(component_df.sort_values("dna", ascending=False).head(10))
# %%
number_logits, number_cache = model.run_with_cache(tokens)
number_cache: ActivationCache
resid_stack, resid_labels = number_cache.get_full_resid_decomposition(layer=8, mlp_input=True, apply_ln=True, return_labels=True, expand_neurons=False)
dna_stack_9 = resid_stack[:, :, -1] @ win

line([dna_stack_9[:, :].mean(1)], line_labels=["top", "not top"], title="DNA for dataset example 9 and neuron L8N1676", x=resid_labels)
component_df = pd.DataFrame({"label":resid_labels, "dna":to_numpy(dna_stack_9.mean(1))})
nutils.show_df(component_df.sort_values("dna", ascending=False).head(10))
# %%
px.box(to_numpy(number_cache["pattern", 6][:, 1, -1]))#, x=nutils.process_tokens_index(prompts[0]))
# %%
resid_stack, resid_labels = number_cache.get_full_resid_decomposition(layer=6, mlp_input=False, apply_ln=True, return_labels=True, expand_neurons=False, pos_slice=9)
line(((resid_stack @ model.W_V[6, 1] @ model.W_O[6, 1] @ win) / number_cache["scale", 8, "ln2"][:, -1, 0]).mean(1), x=resid_labels, title="Effect mediated by L6H1")
# %%
resid_stacks = [(number_cache["post", i][:, -3, :] / number_cache["scale", 8, "ln2"][:, -1, :]).mean(0) * (model.W_out[i] @ model.W_V[6, 1] @ model.W_O[6, 1] @ win) for i in [0, 3, 4, 5]]
line(resid_stacks, title="Effect mediated by L6H1", line_labels=["L0", "L3", "L4", "L5"])
# %%
x = model.W_E / number_cache["scale", 0, "ln2"][:, -3, 0].mean() @ model.W_in[0, :, 908]
nutils.show_df(nutils.create_vocab_df(x).head(30))
is_digit = [s.strip().isdigit() for s in model.to_str_tokens(np.arange(d_vocab))]
px.histogram(to_numpy(x), color=is_digit, marginal="box", barmode="overlay", hover_name=model.to_str_tokens(np.arange(d_vocab)), title='L0N908 embedding connections')
# %%
resid_stack, resid_labels = local_cache.get_full_resid_decomposition(layer=7, mlp_input=False, apply_ln=True, return_labels=True, expand_neurons=False)
line(((resid_stack[:, 0, top_acts, :] @ model.W_V[7, 11] @ model.W_O[7, 11] @ win) / number_cache["scale", 8, "ln2"][:, -1, 0]).mean(1), x=resid_labels, title="Effect mediated by L7H11")
# %%
imshow(((model.W_out[4, 364, :][None, :] @ model.W_V[1:9]) @ (model.W_O[1:9] @ model.W_in[8, :, 1676][:, None])).squeeze(), xaxis="Head", yaxis="Layer", y=np.arange(1, 9))
# %%
resid_stacks = [(number_cache["post", i][:, -1, :] / number_cache["scale", 8, "ln2"][:, -1, :]).mean(0) * (model.W_out[i] @ model.W_V[6, 1] @ model.W_O[6, 1] @ win) for i in [0, 3, 4, 5]]
line(resid_stacks, title="Effect mediated by L6H1", line_labels=["L0", "L3", "L4", "L5"])
# %%
is_digit = np.array([s.strip().isdigit() for s in model.to_str_tokens(np.arange(d_vocab))])
is_two_digit = np.array([s.strip().isdigit() and len(s.strip())==2 for s in model.to_str_tokens(np.arange(d_vocab))])
ave_digit_U = model.W_U[:, is_digit].mean(-1)
ave_not_digit_U = model.W_U[:, ~is_digit].mean(-1)
diff_digit_U = ave_digit_U - ave_not_digit_U
line(model.W_out @ diff_digit_U)
neuron_df["diff_digit_U"] = to_numpy((model.W_out[9:11] @ diff_digit_U).flatten())

neuron_df["digit_std"] = to_numpy((model.W_out[9:11] @ model.W_U[:, is_digit]).std(-1).flatten())
neuron_df["2_digit_std"] = to_numpy((model.W_out[9:11] @ model.W_U[:, is_two_digit]).std(-1).flatten())
px.scatter(neuron_df, x="diff_digit_U", y="digit_std", facet_col="L").show()
px.scatter(neuron_df, x="digit_std", y="2_digit_std", facet_col="L").show()
# %%

neuron_df["cos"] = to_numpy((wout_norm @ (model.W_in[9:11] / model.W_in[9:11].norm(dim=1, keepdim=True))).flatten())
px.scatter(neuron_df, x="cos", y="diff_digit_U", trendline="ols", hover_name="N", facet_col="L").show()
px.scatter(neuron_df, x="cos", y="digit_std", trendline="ols", hover_name="N", facet_col="L").show()
# %%
number_tokens = torch.tensor([model.to_single_token(str(100+x)[1:]) for x in range(1, 100)])
line(number_tokens)

prompts = [f"The war was fought from the year {x} to 16" for x in range(1605, 1695)]
tokens = model.to_tokens(prompts)
base_logits, cache = model.run_with_cache(tokens)
base_log_probs = base_logits[:, -1, :].log_softmax(dim=-1)
imshow(-base_log_probs[:, number_tokens])
number_base_log_probs = -base_log_probs[:, number_tokens]

def get_metrics(patched_logits):
    if len(patched_logits.shape)==3:
        patched_logits = patched_logits[:, -1, :]
    patched_log_probs = patched_logits.log_softmax(dim=-1)
    patched_probs = patched_logits.softmax(dim=-1)
    
    number_probs = patched_probs[:, number_tokens]
    prob_above = torch.where(np.arange(5, 95)[:, None]<np.arange(1, 100)[None, :], number_probs, torch.tensor(0.)).sum(-1) / (np.arange(5, 95)[:, None]<np.arange(1, 100)[None, :]).sum(-1)
    prob_below = torch.where(np.arange(5, 95)[:, None]>np.arange(1, 100)[None, :], number_probs, torch.tensor(0.)).sum(-1) / (np.arange(5, 95)[:, None]>np.arange(1, 100)[None, :]).sum(-1)
    prob_diff = prob_above.mean() - prob_below.mean()
    prob_above = prob_above.mean()
    prob_below = prob_below.mean()

    number_logits = patched_logits[:, number_tokens]
    pm1_logit_diff = number_logits[np.arange(len(number_logits)), np.arange(5, 95)+1] - number_logits[np.arange(len(number_logits)), np.arange(5, 95)-1]
    pm2_logit_diff = number_logits[np.arange(len(number_logits)), np.arange(5, 95)+2] - number_logits[np.arange(len(number_logits)), np.arange(5, 95)-2]

    kl_div = patched_
