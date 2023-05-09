import functools as ft
import itertools as it
import os
import sys
from typing import NamedTuple, Optional

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import haiku as hk
import haiku.initializers as hki
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax
import polars as pl
import rich.progress as rp

jax.config.update("jax_debug_nans", True)

tags = (
    pl.scan_csv("tags.csv")
    .with_columns(pl.col("post_count").arg_sort())
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
        .groupby("id")
    )
    shard = (
        shard.join(joint.agg({"id_tag": "first"}), on="id")
        .with_columns(pl.col("id_tag").arr.sort().arr.slice(0, topk))
        .with_columns(pl.col("age").fill_null(strategy="forward"))
        .select(Batch._fields)
    )
    if seed is not None:
        shard = shard.sample(len(shard), shuffle=True, seed=seed)
    tag_ids = np.zeros([len(shard), topk], dtype=np.int32)
    for j, row in enumerate(shard["id_tag"]):
        tag_ids[j, : min(len(row), topk)] = row
    return Batch(*shard.drop("id_tag").to_numpy().T.astype(float), tag_ids)


class DCN(hk.Module):
    "https://arxiv.org/abs/2008.13535"

    def __init__(self, ranks, name=None):
        super().__init__(name=name)
        self.ranks = ranks

    def __call__(self, x):
        y = x
        for d in self.ranks:
            U = hk.Linear(d)
            Vt = hk.Linear(x.shape[-1])
            y += x * Vt(U(y))
        return y


def regressor(batch: Batch, dropout=0.1):
    def mueller_hash(x):
        "https://stackoverflow.com/a/12996028"
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x = (x >> 16) ^ x
        return x

    rng = hk.maybe_next_rng_key()
    if rng is not None:
        rng = hk.PRNGSequence(rng)
    width = 256
    embed = hk.Embed(4096, width, w_init=hki.RandomNormal(width**-0.5))
    index = batch.id_tag
    z = jnp.zeros(batch.id_tag.shape[:-1] + (embed.embed_dim,))
    for _ in range(4):
        "https://arxiv.org/abs/1706.03993"
        "https://explosion.ai/blog/bloom-embeddings"
        index += embed.vocab_size
        z += embed(mueller_hash(index % embed.vocab_size)).mean(axis=-2)
    x = [
        batch.age,
        batch.rating,
        # batch.tag_count,
        # batch.comment_count,
        # batch.up_score,
        # batch.down_score,
    ]
    x = jnp.stack(x, axis=-1)
    r = jax.nn.one_hot(batch.rating, 3)
    "https://arxiv.org/abs/1907.05321"
    t = jnp.sin(hk.Linear(width)(batch.age[..., None].astype(float)))
    x = jnp.concatenate([z, x, t, r], axis=-1)
    x += hk.LayerNorm(-1, True, True, scale_init=jnp.zeros)(
        DCN([16] * 4)(hk.dropout(next(rng), dropout, x) if rng is not None else x)
    )
    y = hk.nets.MLP([512, 1024, 512, 2], activation=jax.nn.gelu)(
        x,
        dropout_rate=dropout if rng is not None else None,
        rng=next(rng) if rng is not None else None,
    )
    shift, scale = y.swapaxes(0, -1)
    scale = jax.nn.softplus(scale)
    return shift, scale


@hk.transform
def gaussian_nll(batch: Batch):
    shift, scale = regressor(batch)
    return -jnp.mean(jax.scipy.stats.norm.logpdf(batch.fav_count, shift, scale))


@hk.transform
def residual_err(batch: Batch):
    shift, scale = regressor(batch)
    del scale
    return jnp.mean(jnp.abs(batch.fav_count - shift))


@hk.transform
def quantile_err(batch: Batch, q=0.32):
    "https://hackmd.io/@cgarciae/quantile-regression"
    shift, scale = regressor(batch)
    del scale
    r = batch.fav_count - shift
    return jnp.maximum(q * r, (q - 1.0) * r).mean()


class EMA(NamedTuple):
    "https://blog.fugue88.ws/archives/2017-01/The-correct-way-to-start-an-Exponential-Moving-Average-EMA"
    r: float
    s: float
    d: float

    @classmethod
    def init(cls, r):
        return cls(r=r, s=0, d=1)

    def update(self, x):
        s = self.r * self.s + (1 - self.r) * x
        d = self.r * self.d
        s /= 1 - d
        return self._replace(r=self.r, s=s, d=d)


def optimizer(tsteps, rng, batch):
    del rng, batch  # needed for SAM ascent step, but otherwise unnecessary
    lsched = optax.linear_onecycle_schedule(tsteps, 3e-5)
    msched = optax.linear_onecycle_schedule(
        tsteps, 0.95, div_factor=0.95 / 0.85, final_div_factor=1
    )
    optim = optax.chain(
        optax.adaptive_grad_clip(1e-3),
        optax.inject_hyperparams(optax.scale_by_lion)(msched),
        optax.additive_weight_decay(0.3),
        optax.inject_hyperparams(optax.scale)(lsched),
        optax.scale(-1),
    )
    return optax.lookahead(optim, 6, 0.5)


class TrainState(NamedTuple):
    params: optax.LookaheadParams
    opt_st: optax.OptState
    loss: EMA


def train_init(steps: int, rng: jrd.PRNGKey, batch: Batch):
    params = hk.transform(regressor).init(rng, batch)
    params = optax.LookaheadParams.init_synced(params)
    opt_st = optimizer(steps, rng, batch).init(params)
    return TrainState(params, opt_st, loss=EMA.init(0.9))


objective = gaussian_nll


@ft.partial(jax.jit, static_argnums=0)
def train_step(steps, state: TrainState, rng: jrd.PRNGKey, batch: Batch):
    grad = jax.grad(objective.apply)(state.params.fast, rng, batch)
    "https://arxiv.org/abs/2102.11600"
    ascent_stride = 1.0 / optax.global_norm(grad)
    params = jax.tree_util.tree_map(
        lambda w, dw: w + jnp.square(w) * dw * ascent_stride, state.params.fast, grad
    )
    loss, grad = jax.value_and_grad(objective.apply)(params, rng, batch)
    updates, opt_st = optimizer(steps, rng, batch).update(
        grad, state.opt_st, state.params
    )
    params = optax.apply_updates(state.params, updates)
    loss = state.loss.update(loss)
    return state._replace(params=params, opt_st=opt_st, loss=loss)


ckpt_path = "params.msgpack"
post_path = "posts.csv"
keys = hk.PRNGSequence(42)
batch_size = 4096
tag_count = 128
sys.stderr.write(f"read {post_path}...\n")
posts = (
    pl.scan_csv(post_path, try_parse_dates=True, low_memory=True)
    .filter(pl.col("is_deleted") == "f")
    .with_columns(pl.col("created_at").fill_null(strategy="backward"))
    .with_columns(
        ((pl.col("created_at") - pl.date(2007, 2, 10)).dt.hours())
        .alias("age")
        .cast(float),
        pl.col("up_score", "comment_count", "down_score", "fav_count").cast(float),
        (pl.col("tag_string").str.count_match(" ") + 1).alias("tag_count").cast(float),
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
    with open(ckpt_path, "rb") as istrm:
        params = msgpack.unpackb(istrm.read())
else:
    tstate = train_init(tsteps, next(keys), next(batches))
    with rp.Progress(
        "loss: {task.fields[loss]:.3g}", *columns, console=console
    ) as pbar:
        task = pbar.add_task(
            "training...",
            loss=float("nan"),
            total=tsteps,
            start=False,
        )
        for rng, batch in zip(keys, batches):
            pbar.start_task(task)
            tstate = train_step(tsteps, tstate, rng, batch)
            pbar.update(task, advance=1, loss=jax.device_get(tstate.loss.s))
    params = jax.device_get(tstate.params.slow)
    with open(ckpt_path, "wb+") as ostrm:
        ostrm.write(msgpack.dumps(params))


@hk.without_apply_rng
@hk.transform
def forward(inputs: Batch):
    value = inputs.fav_count
    shift, scale = regressor(inputs)
    if objective is gaussian_nll:
        score = jax.scipy.stats.norm.cdf(value, shift, scale)
    else:
        del scale
        score = value - shift
    return score


header = ["id", "fav_score"]
sys.stdout.buffer.write((",".join(header) + "\n").encode("utf8"))
with rp.Progress(*columns, console=console, redirect_stdout=False) as pbar:
    task = pbar.add_task(
        "evaluating...", total=np.ceil(len(posts) / batch_size).astype(int)
    )
    for shard in posts.iter_slices(batch_size):
        batch = shard_batch(shard, tag_count)
        score = jax.device_get(jax.jit(forward.apply)(params, batch))
        shard = shard.with_columns(pl.Series(score).alias("fav_score"))
        shard = shard.select(header)
        shard.write_csv(sys.stdout.buffer, has_header=False)
        pbar.advance(task)
