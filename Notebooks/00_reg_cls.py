#!/usr/bin/env python
# coding: utf-8

from tqdm.auto import tqdm

import jax
from jax import random, numpy as jnp
from flax import nnx
import optax


from datasets import load_dataset


dst = load_dataset("Jorgvt/TID2008")


def obtain_dmos(sample):
    sample["dmos"] = (10 - sample["mos"])/10
    return sample


dst = dst.map(obtain_dmos)


dst_train = dst["train"].with_format("jax")


def preprocess(row, resize_to=None):
    img, dist, mos, dmos = row.values()
    img = img/255.
    dist = dist/255.
    if resize_to is not None:
        img = jax.image.resize(img, shape=resize_to, method="linear")
        dist = jax.image.resize(img, shape=resize_to, method="linear")
    return {"reference": img, "distorted": dist, "mos": mos, "dmos": dmos}


# dst_train_rdy = dst_train.map(lambda x: preprocess(x, resize_to=(256,256,3)), num_proc=8)


def pearson_correlation(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    num = vec1 - vec1_mean
    num *= vec2 - vec2_mean
    num = num.sum()
    denom = jnp.sqrt(jnp.sum((vec1 - vec1_mean) ** 2))
    denom *= jnp.sqrt(jnp.sum((vec2 - vec2_mean) ** 2))
    return num / denom


def mse(a, b):
    return ((a-b)**2).mean()**(1/2)


class Model(nnx.Module):
    def __init__(self, *, rngs):
        self.layers = [
            nnx.Conv(in_features=3, out_features=32, kernel_size=5, padding="SAME", rngs=rngs),
            nnx.Conv(in_features=32, out_features=64, kernel_size=3, padding="SAME", rngs=rngs),
            nnx.Conv(in_features=64, out_features=128, kernel_size=3, padding="SAME", rngs=rngs),
        ]
        self.sigma_c = nnx.Param((1.))
        self.sigma_r = nnx.Param((1.))

    def __call__(self, inputs, **kwargs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
            outputs = nnx.relu(outputs)
            outputs = nnx.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        return outputs, self.sigma_c, self.sigma_r


@nnx.jit
def train_step(model, optimizer, img, dist, mos, **kwargs):
    def loss_fn(model, **kwargs):
        img_pred, sigma_c, sigma_r = model(img)
        dist_pred, sigma_c, sigma_r = model(dist)
        # sigma_c, sigma_r = jnp.exp(sigma_c), jnp.exp(sigma_r)
        distance = ((img_pred-dist_pred)**2).mean(axis=(1,2,3))**(1/2)
        ## Calculate losses
        corr = 1 - pearson_correlation(distance, mos) # We are using the dmos, so the corr is positive
        reg = mse(distance, mos)
        adding_term = sigma_c + sigma_r
        # return (1/sigma_c**2)*corr + (1/sigma_r**2)*reg + jnp.log(sigma_c) + jnp.log(sigma_r), (corr, reg)
        return jnp.exp(-sigma_c)*corr + jnp.exp(-sigma_r)*reg + adding_term, (corr, reg, adding_term)
    (loss, (corr, reg, adding_term)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return loss, corr, reg, adding_term


model = Model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


def preprocess(row, resize_to=None):
    img, dist, mos, dmos = row.values()
    img = img/255.
    dist = dist/255.
    if resize_to is not None:
        img = jax.image.resize(img, shape=resize_to, method="linear")
        dist = jax.image.resize(dist, shape=resize_to, method="linear")
    return img, dist, dmos


epochs = 50
losses = []
losses_c, losses_r, losses_add = [], [], []
for epoch in tqdm(range(epochs), desc="Epoch"):
    losses_b, losses_b_c, losses_b_r, losses_b_add = [], [], [], []
    for batch in tqdm(dst_train.iter(batch_size=32, drop_last_batch=True), leave=False, desc="Batches"):
        img, dist, mos = preprocess(batch, resize_to=(32,256,256,3))
        loss, loss_c, loss_r, loss_add = train_step(model, optimizer, img, dist, mos)
        losses_b.append(loss)
        losses_b_c.append(loss_c)
        losses_b_r.append(loss_r)
        losses_b_add.append(loss_add)
        # break
    losses.append(jnp.mean(jnp.array(losses_b)))
    losses_c.append(jnp.mean(jnp.array(losses_b_c)))
    losses_r.append(jnp.mean(jnp.array(losses_b_r)))
    losses_add.append(jnp.mean(jnp.array(losses_b_add)))
    print(f"Epoch {epoch} --> Loss: {losses[-1]} | Corr: {losses_c[-1]} ({jnp.exp(model.sigma_c.value):.2f})| Reg: {losses_r[-1]} ({jnp.exp(model.sigma_r.value):.2f}) | Adding: {losses_add[-1]}")
    # break




