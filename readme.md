# E6Score

Quick and dirty estimates of logged implicit feedback from tabular metadata present on E621.

This project isn't intended for standalone use, but rather for counterfactual reasoning about how much of the implicit feedback on a post has been influenced by factors other than its technical execution, such as to mitigate those factors and yield a more faithful estimate of the score as explained by the quality of the post. E.g., during Monte Carlo sampling of logged feedback to train a downstream (hopefully smarter) model: `E[score | metadata, aesthetics] - E[score | metadata] ≈ E[score | aesthetics]`, s.t. the improved estimator considers only whether a post's observed ratings over- or under-performed expectations given its tags and a variety of other housekeeping information. The estimate is reweighted by the model's confidence in it when the final 'adjusted' rating is produced. Note that the actual underlying aesthetics of the post are not considered, only its metadata. Read the source for more details.

## Running it

First, you need your dependencies. Run:
```
$ pip install -r requirements.txt
```

Second, execute

```
$ ./fetch.sh
```

This just grabs data to fit the model. 

Finally, you can run the code. During this step, if you installed a jaxlib wheel compiled without GPU support in step 1, XLA may complain that it cannot access your GPU. The model topology is simple enough and its parameterization is lightweight enough that this does not matter. Throughput will be high enough to get through the entire data dump in a reasonable amount of time, processor notwithstanding. Might even be faster than using SPMD accelerators enumerated on the bus, since the overhead of executing this model is dominated simply by data moving back and forth over that bus, rather than compute happening on the device. 

```
$ python3 score.py > score.csv # or py.exe if you're under Windows and still have a half decent build of jax somehow
```

This will train the regressor to maximize likelihood (minimize negative log-likelihood) of observed feedback by backpropagating through the parameters of a Gaussian distribution, and output the results of running inference to a new or truncated file called `score.csv` that passes through all the old columns, and adds a new one called `fav_score` (overwriting the previous result). It will also create (or truncate) a `params.msgpack` file that checkpoints the parameters found in the learning phase, so you will have to manually clear this cache by deleting the file if you want to fit to a new `posts.csv`. After that, whatever you want to use it for is all up to your curation goals or analytical curiosities. Good hunting!

