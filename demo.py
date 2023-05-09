from urllib.parse import urlencode

import polars as pl

pool_ids = (
    pl.scan_csv("pools.csv")
    .select(pl.col("post_ids").str.strip("{}").str.split(","))
    .explode("post_ids")
    .unique()
    .filter(pl.col("post_ids") != "")
    .select(pl.col("post_ids").cast(int))
    .collect()
    .to_series()
)
k = 32
ids = (
    pl.scan_csv("ratings.csv")  # or wherever you piped yours
    .join(pl.scan_csv("posts.csv"), on="id")
    .filter(~pl.col("id").is_in(pool_ids))
    .filter(~(pl.col("tag_string").str.contains(r"humor|meme|comic")))
    .filter(pl.col("file_ext").is_in(["png", "jpg", "webp"]))
    .filter(pl.col("rating") == "s")
    .sort(pl.col("fav_score") * pl.col("fav_count"))
    .select("id")
    .collect()[-k:]
    .to_series()
)
print("https://e621.net/posts?" + urlencode({"tags": f"id:" + ",".join(ids.cast(str))}))
