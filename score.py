import functools as ft
import itertools as it
import os
import sys
from typing import NamedTuple, Optional
import orbax.checkpoint

import flax.linen as nn
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax
import polars as pl
import rich.progress as rp
import jax.tree_util as jtu

tags = (
    pl.scan_csv("tags.csv")
    .with_columns(pl.col("post_count").arg_sort().alias("id"))
    .filter(pl.col("post_count") > 0)
    .collect()
)


class Batch(NamedTuple):
    age: jnp.ndarray
    rating: jnp.ndarray
    fav_count: jnp.ndarray
    tag_count: jnp.ndarray
    up_score: jnp.ndarray
    down_score: jnp.ndarray
    comment_count: jnp.ndarray
    id_tag: jnp.ndarray


def preprocess(shard: pl.DataFrame):
    return shard.with_columns(
        pl.col("tag_string").str.split(" ").alias("tags")
    ).with_columns(pl.col("rating").cast(pl.Categorical).cast(pl.Int8))


def shard_batch(shard: pl.DataFrame, topk: int, seed: Optional[int] = None):
    shard = preprocess(shard).select(["tags", "id", *Batch._fields[:-1]])
    joint = (
        shard.select(["tags", "id"])
        .explode("tags")
        .join(tags, left_on="tags", right_on="name", how="left", suffix="_tag")
        .group_by("id")
    )
    shard = (
        shard.join(joint.agg(pl.col("id_tag")), on="id")
        .with_columns(pl.col("id_tag").list.sort().list.slice(0, topk))
        .with_columns(pl.col("age").fill_null(strategy="forward"))
        .select(Batch._fields)
    )
    if seed is not None:
        shard = shard.sample(len(shard), shuffle=True, seed=seed)
    tag_ids = np.zeros([len(shard), topk], dtype=np.int32)
    for j, row in enumerate(shard["id_tag"]):
        tag_ids[j, : min(len(row), topk)] = row
    return Batch(*shard.drop("id_tag").to_numpy().T.astype(float), tag_ids)


class DCN(nn.Module):
    "https://arxiv.org/abs/2008.13535"
    ranks: list[int]

    @nn.compact
    def __call__(self, x):
        y = x
        for d in self.ranks:
            U = nn.Dense(d)
            Vt = nn.Dense(x.shape[-1])
            y += x * Vt(U(y))
        return y


class MLP(nn.Module):
    ranks: list[int]
    actfn: nn.activation = nn.gelu

    @nn.compact
    def __call__(self, x):
        _act = self.actfn
        y = x
        for i, d in enumerate(self.ranks):
            y = nn.Dense(d)(y)
            if i != len(self.ranks) - 1:
                y = _act(y)
        return y


class Regressor(nn.Module):
    width: int = 256
    vocab: int = 4096
    dcn_ranks: tuple[int] = (16,) * 4
    mlp_ranks: tuple[int] = (512, 68, 382)
    emb_steps: int = 4

    @nn.compact
    def __call__(self, batch):
        def mueller_hash(x):
            "https://stackoverflow.com/a/12996028"
            x = ((x >> 16) ^ x) * 0x45D9F3B
            x = ((x >> 16) ^ x) * 0x45D9F3B
            x = (x >> 16) ^ x
            return x

        width = self.width
        embed = nn.Embed(self.vocab, width)
        index = batch.id_tag
        z = jnp.zeros(batch.id_tag.shape[:-1] + embed.embedding.shape[-1:])
        for embed_step in range(self.emb_steps):
            "https://arxiv.org/abs/1706.03993"
            "https://explosion.ai/blog/bloom-embeddings"
            index = mueller_hash(index)
            if embed_step % 2 == 1:
                index = ~index
            codes = embed(index % embed.num_embeddings)
            codes *= self.emb_steps**-0.5
            codes *= codes.shape[-2] ** -0.5
            codes = codes.sum(axis=-2)
            z += codes

        r = nn.Embed(3, width)(batch.rating.astype(int))
        "https://arxiv.org/abs/1907.05321"
        t = jnp.sin(nn.Dense(width)(batch.age[..., None].astype(float)))
        x = jnp.concatenate([z, t, r], axis=-1)
        x += DCN(self.dcn_ranks)(x)
        y = MLP(self.mlp_ranks + (1,), actfn=jax.nn.gelu)(x)
        y = y.squeeze(axis=-1)
        return y


def flatten(params):
    arrays = jtu.tree_leaves(params)
    assert all(a.dtype == arrays[0].dtype for a in arrays[1:])
    return jnp.concatenate(
        [param.reshape(-1) for param in jtu.tree_leaves(params)],
        dtype=arrays[0].dtype,
    )


def unflatten(flat, updates):
    updates_flat, treedef = jtu.tree_flatten(updates)
    offsets = []
    for update in updates_flat:
        if offsets:
            offsets.append(update.size + offsets[-1])
        else:
            offsets.append(update.size)
    del offsets[-1]
    flat_split = jnp.split(flat, offsets)
    reshaped = [
        jnp.reshape(flat_update, update.shape)
        for flat_update, update in zip(flat_split, updates_flat)
    ]
    return jtu.tree_unflatten(treedef, reshaped)


class TrainState(NamedTuple):
    params: optax.Params
    scales: jax.Array
    moment: jax.Array
    # opt_st: optax.OptState
    err_st: optax.EmaState
    loss: jax.Array
    step: jax.Array
    rng: jrd.PRNGKey


@ft.partial(jax.jit, static_argnums=0, donate_argnums=1)
def train_init(steps: int, rng: jrd.PRNGKey, inputs: Batch):
    rng, key = jrd.split(rng)
    params = Regressor().init(key, inputs)
    # params["alpha"] = jnp.zeros([])
    # opt_st = optimizer(steps).init(params)
    scales = jnp.ones_like(flatten(params))
    moment = jnp.zeros_like(scales)
    loss = 0.0
    err_st = optax.ema(0.9).init(loss)
    step = 0
    return TrainState(params, scales, moment, err_st, loss, step, rng)


def quantile_err(params, inputs: Batch, q=0.99):
    "https://hackmd.io/@cgarciae/quantile-regression"
    r = inputs.fav_count - Regressor().apply(params, inputs)
    return jnp.maximum(q * r, (q - 1.0) * r).mean()


def objective(params, inputs):
    y = inputs.fav_count
    # alpha = params.pop("alpha")
    y_hat = Regressor().apply(params, inputs)
    r_err = optax.l2_loss(y, y_hat)
    h_err = jnp.abs(y_hat.var() - jnp.ones_like(y_hat))
    error = r_err.mean()
    # error = (
    #     jax.nn.sigmoid(alpha) * r_err.mean()
    #     + (1 - jax.nn.sigmoid(alpha)) * h_err.mean()
    # )
    # params["alpha"] = alpha
    return error


# objective = quantile_err


def optimizer(steps):
    peak_lr = 1e-3
    weight_decay = 0.01
    lsched = optax.linear_onecycle_schedule(steps, peak_lr)
    msched = lambda step: 0.85 + 0.1 * lsched(step) / peak_lr
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.additive_weight_decay(weight_decay),
        optax.inject_hyperparams(optax.scale_by_lion)(msched),
        optax.inject_hyperparams(optax.scale)(lsched),
    )
    return optax.flatten(optim)


@ft.partial(jax.jit, static_argnums=0)
def train_step(steps, state: TrainState, inputs: Batch):
    peak_lr = 1e-3
    lsched = optax.linear_onecycle_schedule(steps, peak_lr)
    msched = lambda step: 0.85 + 0.1 * lsched(step) / peak_lr
    ascent_stride = 0.01
    weight_decay = 0.00
    scales_gamma = 1e-1
    clipping_factor = 1.0
    batch_size = inputs.fav_count.size
    params = flatten(state.params)
    rng, ascent_key = jrd.split(state.rng)
    _unflatten = ft.partial(unflatten, updates=state.params)
    ascent = jrd.normal(ascent_key, params.shape, dtype=params.dtype) * (
        1 / jnp.sqrt(state.scales) / batch_size
    )
    grad = jax.grad(objective)(_unflatten(params + ascent), inputs)
    grad = flatten(grad)
    scales = state.scales
    ascent = ascent_stride * grad / scales
    scales = jnp.sqrt(scales) * jnp.abs(grad) + weight_decay + scales_gamma
    scales = optax.update_moment(grad, scales, 0.99, order=1)
    grad = jax.grad(objective)(_unflatten(params + ascent), inputs)
    grad = flatten(grad)
    loss, grad = jax.value_and_grad(objective)(_unflatten(params + ascent), inputs)
    step = optax.safe_int32_increment(state.step)
    # grad["alpha"] *= 0.01 / -lsched(step)
    grad = flatten(grad)
    grad /= optax.safe_norm(grad, 1e-5) * clipping_factor
    moment = state.moment
    moment = optax.update_moment(grad, moment, msched(state.step), order=1)
    params -= (
        lsched(step)
        * optax.bias_correction(moment, msched(step), step)
        / optax.bias_correction(scales, 0.99, step)
        + weight_decay * params
    )
    params = _unflatten(params)
    loss, err_st = optax.ema(0.9).update(loss, state.err_st)
    return state._replace(
        params=params,
        err_st=err_st,
        moment=moment,
        scales=scales,
        step=step,
        loss=loss,
        rng=rng,
    )


ckpt_path = "params.ckpt"
post_path = "posts.csv"
batch_size = 4096
tag_count = 32
sys.stderr.write(f"read {post_path}...\n")
posts = (
    pl.scan_csv(post_path)
    .with_columns(
        pl.col("^(created|updated)_at$")
        .str.split(".")
        .list.get(0)
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    .filter(pl.col("is_deleted") == "f")
    .with_columns(pl.col("created_at").fill_null(strategy="backward"))
    .with_columns(
        ((pl.col("created_at") - pl.date(2007, 2, 10)).dt.hours())
        .alias("age")
        .cast(float),
        pl.col("up_score", "comment_count", "down_score", "fav_count").cast(float),
        (pl.col("tag_string").str.count_matches(" ") + 1)
        .alias("tag_count")
        .cast(float),
    )
    .with_columns(
        (pl.col(pl.FLOAT_DTYPES) - pl.col(pl.FLOAT_DTYPES).median())
        / pl.col(pl.FLOAT_DTYPES).std()
    )
    .collect()
)
epochs = 1.0
tsteps = np.ceil(epochs * len(posts) / batch_size).astype(int)
breaks = np.arange(tsteps)
ckpointer = orbax.checkpoint.PyTreeCheckpointer()
batches = (
    shard_batch(posts.sample(batch_size, seed=seed), tag_count) for seed in it.count()
)
batches = it.islice(batches, tsteps)
columns = (
    *rp.Progress.get_default_columns()[:-2],
    rp.MofNCompleteColumn(),
    rp.TimeElapsedColumn(),
)
console = rp.Console(file=sys.stderr)
if os.path.exists(ckpt_path):
    params = ckpointer.restore(ckpt_path)
else:
    rng = jrd.PRNGKey(42)
    tstate = train_init(tsteps, rng, next(batches))
    shards = jax.sharding.PositionalSharding(jax.local_devices())
    tstate = jax.device_put(tstate, shards.replicate(0))
    with rp.Progress(
        "loss: {task.fields[loss]:.3g}", *columns, console=console
    ) as pbar:
        task = pbar.add_task(
            "training...",
            loss=float("nan"),
            total=tsteps,
            start=False,
        )
        for inputs in batches:
            inputs = jtu.tree_map(
                lambda a: jax.device_put(a, shards.reshape([-1] + [1] * (a.ndim - 1))),
                inputs,
            )
            tstate = train_step(tsteps, tstate, inputs)
            pbar.start_task(task)
            pbar.update(task, advance=1, loss=jax.device_get(tstate.loss))
    params = jax.device_get(tstate.params)
    ckpointer.save(
        ckpt_path, params, save_args=orbax_utils.save_args_from_target(params)
    )


header = ["id", "fav_score"]
sys.stdout.buffer.write((",".join(header) + "\n").encode("utf8"))
stddev = posts.select(pl.col("fav_count").std())
with rp.Progress(*columns, console=console, redirect_stdout=False) as pbar:
    task = pbar.add_task(
        "evaluating...", total=np.ceil(len(posts) / batch_size).astype(int)
    )
    # alpha = params.pop("alpha")
    for shard in posts.iter_slices(batch_size):
        inputs = shard_batch(shard, tag_count)
        value = jax.device_get(jax.jit(Regressor().apply)(params, inputs))
        score = (inputs.fav_count - value) / stddev
        shard = shard.with_columns(pl.Series("fav_score", value))
        shard = shard.select(header)
        shard.write_csv(sys.stdout.buffer, has_header=False)
        pbar.advance(task)
