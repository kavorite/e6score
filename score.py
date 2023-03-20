import functools as ft
import itertools as it
import os
import sys
from typing import NamedTuple

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
    is_deleted: jnp.ndarray
    id_tag: jnp.ndarray


def preprocess(shard: pl.DataFrame):
    return (
        shard.with_columns(pl.col("tag_string").str.split(" ").alias("tags"))
        .with_columns(pl.col("rating").cast(pl.Categorical).cast(int))
        .with_columns(pl.col("is_deleted") == "t")
        .with_columns(pl.col("tags").arr.lengths().alias("tag_count"))
        .with_columns(pl.col("created_at").fill_null(strategy="backward"))
        .with_columns(
            (pl.col("created_at") - pl.date(2007, 2, 10)).dt.days().alias("age")
        )
    )


def shard_batch(shard: pl.DataFrame, topk: int):
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
    tag_ids = np.zeros([len(shard), topk], dtype=np.int32)
    for j, row in enumerate(shard["id_tag"].to_numpy()):
        tag_ids[j, : min(len(row), topk)] = row
    return Batch(*shard.to_numpy().swapaxes(-1, 0)[:-1].astype(float), tag_ids)


class DCN(hk.Module):
    "https://arxiv.org/abs/2008.13535"

    def __init__(self, ranks, name=None):
        super().__init__(name=name)
        self.ranks = ranks

    def __call__(self, x):
        y = x
        for d in self.ranks:
            y += x * hk.Linear(x.shape[-1])(hk.Linear(d)(y))
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
        "https://explosion.ai/blog/bloom-embeddings"
        index = mueller_hash(index)
        z += embed(index % embed.vocab_size).mean(axis=-2)
    x = [
        batch.age,
        batch.tag_count,
        batch.rating,
        batch.is_deleted,
        batch.comment_count,
        batch.up_score,
        batch.down_score,
    ]
    x = jnp.stack(x, axis=-1)
    r = jax.nn.one_hot(batch.rating, 3)
    x = jnp.concatenate([x, r], axis=-1).astype(float)
    x = hk.LayerNorm(-1, True, True)(x)

    "https://arxiv.org/abs/1907.05321"
    t = jnp.sin(hk.Linear(width)(batch.age[..., None].astype(float)))

    h = jnp.concatenate([z, x, t], axis=-1)
    h = hk.dropout(next(rng), dropout, h) if rng is not None else h
    h = DCN([16] * 4)(hk.LayerNorm(-1, True, True)(h))
    y = hk.nets.MLP([width, 512, 384, 96, 2], activation=jax.nn.gelu)(
        h,
        dropout_rate=dropout if rng is not None else None,
        rng=next(rng) if rng is not None else None,
    )
    shift, scale = y.swapaxes(0, -1)
    return map(jax.nn.softplus, (shift, scale))


@hk.transform
def gaussian_nll(batch: Batch):
    shift, scale = regressor(batch)
    return -jnp.mean(jax.scipy.stats.norm.logpdf(batch.fav_count, shift, scale))


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
    del rng, batch  # needed for SAM, but otherwise unnecessary
    lsched = optax.linear_onecycle_schedule(tsteps, 1e-5)
    msched = optax.linear_onecycle_schedule(
        tsteps, 0.95, div_factor=0.95 / 0.85, final_div_factor=1
    )
    optim = optax.chain(
        optax.inject_hyperparams(optax.trace)(msched),
        optax.inject_hyperparams(optax.scale)(lsched),
        optax.scale(-1),
        optax.clip_by_block_rms(0.01),
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


@ft.partial(jax.jit, static_argnums=0)
def train_step(steps, state: TrainState, rng: jrd.PRNGKey, batch: Batch):
    loss, grad = jax.value_and_grad(gaussian_nll.apply)(state.params.fast, rng, batch)
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
posts = pl.read_csv(post_path, try_parse_dates=True, low_memory=True)
epochs = 1.0
tsteps = int(epochs * len(posts) / batch_size + 0.5)  # bootleg ceil()
breaks = np.arange(0, len(posts) // batch_size)
train_shards = (
    posts.slice(i * batch_size, batch_size)
    for i in jax.jit(jrd.permutation, backend="cpu")(next(keys), breaks)
)
batches = it.islice(
    it.cycle(shard_batch(shard, tag_count) for shard in train_shards), tsteps
)
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
def forward(stddev: float, inputs: Batch):
    value = inputs.fav_count
    shift, scale = regressor(inputs)
    score = (value / stddev + shift / scale) / (1 / stddev + 1 / scale)
    # score = 1 - jax.scipy.stats.norm.cdf(value, shift, scale)
    return score


header = posts.columns + ["fav_score"]
stddev = posts["fav_count"].std()
sys.stdout.buffer.write((",".join(header) + "\n").encode("utf8"))
with rp.Progress(*columns, console=console, redirect_stdout=False) as pbar:
    task = pbar.add_task("evaluating...", total=int(len(posts) / batch_size + 0.5))
    shards = (posts.slice(i * batch_size, batch_size) for i in breaks)
    stddev = posts["fav_count"].std()
    for shard in shards:
        batch = shard_batch(shard, tag_count)
        score = jax.device_get(jax.jit(forward.apply)(params, stddev, batch))
        shard = shard.with_column(pl.Series(score).alias("fav_score"))
        shard = shard.select(header)
        shard.write_csv(sys.stdout.buffer, has_header=False)
        pbar.advance(task)
