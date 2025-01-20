# simple-maps
Simple and effective manifold approximators with few dependencies.

# Spec

## Algorithm

1. Equal to or better runtime than existing implementations at inference time (willing to eat some training cost for large _N_ and dim).
2. UMAP, TriMap, and PACMap.

## Production Ready

1. Device interoperability: GPU, CPU, TPU. Anything goes with little to no thought from the user.
2. Intuitive save and load: straightforward, low-dependency saving and loading. Opinionated towards parametric formulations which are easy to serialize and productionize.
3. One, if any, main dependency as backend e.g. Jax or PyTorch, still unclear. No Annoy/PyNNDescent requirement due to dependency hell.
