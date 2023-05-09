from urllib.parse import urlencode

import polars as pl

k = 32
ids = (
    pl.scan_csv("ratings.csv")  # or wherever you piped yours
    .join(pl.scan_csv("posts.csv"), on="id")
    .filter(pl.col("file_ext").is_in(["jpg", "png", "webp"]))
    .filter(pl.col("rating") == "s")
    .with_columns(pl.col(pl.FLOAT_DTYPES) - pl.col(pl.FLOAT_DTYPES).min())
    .sort(pl.col("fav_score") * pl.col("fav_count"))
    .select("id")
    .collect()[-k:]
    .to_series()
)
print("https://e621.net/posts?" + urlencode({"tags": f"id:" + ",".join(ids.cast(str))}))
