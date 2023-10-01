

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
data = load_dataset("stas/openwebtext-10k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=256)
tokenized_data = tokenized_data.shuffle(42)
batch_size = 32
tokens = tokenized_data[:batch_size]["tokens"]
print(model.to_str_tokens(tokens[0]))
# %%
LAYER = 8
INDEX = 1676
def zero_ablate_neuron(mlp_post, hook):
    mlp_post[:, :, INDEX] = 0.
    return mlp_post
def force_on_neuron(mlp_post, hook):
    mlp_post[:, :, INDEX] = 4.
    return mlp_post


normal_logits, normal_cache = model.run_with_cache(tokens)
normal_clps = model.loss_fn(normal_logits, tokens, True)

model.blocks[LAYER].mlp.hook_post.add_hook(zero_ablate_neuron)
off_logits, off_cache = model.run_with_cache(tokens)
off_clps = model.loss_fn(off_logits, tokens, True)


model.blocks[LAYER].mlp.hook_post.add_hook(force_on_neuron)
on_logits, on_cache = model.run_with_cache(tokens)
on_clps = model.loss_fn(on_logits, tokens, True)

print(normal_clps.mean(), off_clps.mean(), on_clps.mean())
# %%
line(normal_cache["post", LAYER][:, :, INDEX])
# %%
histogram((off_clps - normal_clps).flatten())
histogram((on_clps - normal_clps).flatten())
# %%
orig_token_df = nutils.make_token_df(tokens, 8, 3)
orig_token_df["act"] = to_numpy(normal_cache["post", LAYER][:, :, INDEX]).flatten()
token_df = orig_token_df.query(f"pos < {orig_token_df.pos.max()}")
token_df["next_token"] = nutils.list_flatten([nutils.process_tokens(tokens[i, 1:]) for i in range(batch_size)])
token_df["off_diff"] = to_numpy(off_clps - normal_clps).flatten()
token_df["on_diff"] = to_numpy(on_clps - normal_clps).flatten()
nutils.show_df(token_df.sort_values("on_diff", ascending=False).head(50))
nutils.show_df(token_df.sort_values("on_diff", ascending=False).tail(10))
nutils.show_df(token_df.sort_values("off_diff", ascending=False).head(50))
nutils.show_df(token_df.sort_values("off_diff", ascending=False).tail(10))
# %%
normal_logsumexp = normal_logits.logsumexp(-1)
off_logsumexp = off_logits.logsumexp(-1)
on_logsumexp = on_logits.logsumexp(-1)
off_logsumexp_diff = off_logsumexp - normal_logsumexp
on_logsumexp_diff = on_logsumexp - normal_logsumexp
def plot_hist(off_vals, on_vals, **kwargs):
    off_vals = to_numpy(off_vals.flatten())
    on_vals = to_numpy(on_vals.flatten())
    vals = np.concatenate([off_vals, on_vals])
    color = [False] * len(off_vals) + [True] * len(on_vals)
    temp_df = pd.DataFrame({
        "vals": vals,
        "on": color,
        "context": np.concatenate([orig_token_df.context.values, orig_token_df.context.values], axis=0)
    })
    px.histogram(temp_df, hover_name="context", color="on", barmode="overlay", x="vals", marginal="box", **kwargs).show()
plot_hist(off_logsumexp_diff, on_logsumexp_diff)
# %%
on_corr_logit = on_logits[:, :-1, :].gather(-1, tokens.cuda()[:, 1:, None])
normal_corr_logit = normal_logits[:, :-1, :].gather(-1, tokens.cuda()[:, 1:, None])
px.scatter(x=to_numpy(on_corr_logit - normal_corr_logit).flatten(), y=to_numpy(on_clps - normal_clps).flatten(), trendline="ols", color=to_numpy(on_logsumexp_diff[:, :-1]).flatten(), labels={"x":"logit delta", "y":"log prob delta", "color":"logsumexp delta"}).show()
px.scatter(color=to_numpy(on_corr_logit - normal_corr_logit).flatten(), y=to_numpy(on_clps - normal_clps).flatten(), trendline="ols", x=to_numpy(on_logsumexp_diff[:, :-1]).flatten(), labels={"x":"logsumexp delta", "y":"log prob delta", "color":"logit delta"}).show()
px.scatter(y=to_numpy(on_clps - normal_clps).flatten(), trendline="ols", x=to_numpy(on_corr_logit - normal_corr_logit).flatten()-to_numpy(on_logsumexp_diff[:, :-1]).flatten(), labels={"x":"logsumexp delta", "y":"log prob delta", "color":"logit delta"}).show()
# %%
W_U_tokens = model.W_U[:, tokens[:, 1:]]
W_U_tokens = einops.rearrange(W_U_tokens, "d_model batch pos -> batch pos d_model") / normal_cache["scale"][:, :-1]
print(W_U_tokens.shape)
normal_resid_stack, resid_labels = normal_cache.decompose_resid(apply_ln=False, return_labels=True)
normal_resid_stack = normal_resid_stack[:, :, :-1]
normal_dla = (normal_resid_stack * W_U_tokens).sum(-1).mean([-1, -2])
off_resid_stack, resid_labels = off_cache.decompose_resid(apply_ln=False, return_labels=True)
off_resid_stack = off_resid_stack[:, :, :-1]
off_dla = (off_resid_stack * W_U_tokens).sum(-1).mean([-1, -2])
on_resid_stack, resid_labels = on_cache.decompose_resid(apply_ln=False, return_labels=True)
on_resid_stack = on_resid_stack[:, :, :-1]
on_dla = (on_resid_stack * W_U_tokens).sum(-1).mean([-1, -2])
line([on_dla - normal_dla, off_dla - normal_dla], x=resid_labels)
# %%
def ln_pre(x):
    x = x - x.mean(-1, keepdim=True)
    x = x / (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    return x
wout = model.W_out[LAYER, INDEX]
normal_acts = normal_cache["post", LAYER][:, :, INDEX]
path_patched_mlp11_input = ln_pre(normal_cache["resid_mid", 11] + (4. - normal_acts)[:, :, None] * wout[None, None, :])
patched_mlp11_pre = path_patched_mlp11_input @ model.W_in[11] + model.b_in[11]
patched_mlp11_post = F.gelu(patched_mlp11_pre)
patched_mlp11_out = patched_mlp11_post @ model.W_out[11]
delta_mlp11_out = (normal_cache["mlp_out", 11] - patched_mlp11_out - model.b_out[11])
normal_mlp11_out = normal_cache["mlp_out", 11]
on_mlp11_out = on_cache["mlp_out", 11]
off_mlp11_out = off_cache["mlp_out", 11]
print("Normal", (normal_mlp11_out[:, :-1, :] * W_U_tokens).sum(-1).mean().item())
print("on", (on_mlp11_out[:, :-1, :] * W_U_tokens).sum(-1).mean().item())
print("off", (off_mlp11_out[:, :-1, :] * W_U_tokens).sum(-1).mean().item())
print("patched", ((patched_mlp11_out + model.b_out[11])[:, :-1, :] * W_U_tokens).sum(-1).mean().item())
print("delta", (delta_mlp11_out[:, :-1, :] * W_U_tokens).sum(-1).mean().item())
# %%
delta_mlp11_acts = patched_mlp11_post - normal_cache["post", 11]
W_U_tokens_out = W_U_tokens @ model.W_out[11].T
dla = delta_mlp11_acts[:, :-1, :] * W_U_tokens_out
print(dla.sum(-1).mean())
line(dla.mean([0, 1]))
# %%
vocab_df = nutils.create_vocab_df(model.W_out[11, 611] @ model.W_U)
vocab_df["is_digit"] = vocab_df.token.apply(lambda x: x.split("·")[-1].isdigit())
vocab_df["has_space"] = vocab_df.token.apply(lambda x: "·" in x)
px.histogram(vocab_df, marginal="box", x="logit", color='is_digit', barmode="overlay", hover_name="token").show()
px.histogram(vocab_df, marginal="box", x="logit", color='has_space', barmode="overlay", hover_name="token").show()
# %%
nutils.cos(model.W_out[8, 1676], model.W_in[11, :, 611])
# %%
