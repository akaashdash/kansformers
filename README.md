# Kansformers: Transformers using KANs

[Kolmogorov-Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) are a proposed alternative architecture to MLPs. Despite having similar capabilities (and being representable with MLPs), KANs allow for faster training, less forgetting, and more interpretability.

With some modifications, KANs can be switched out for pytorch nn.Linear layers. We first use [efficient KANs](https://github.com/Blealtan/efficient-kan) and [change shape processing](https://github.com/Blealtan/efficient-kan/pull/6) to allow for better compatability. 

Because the two are interchangable, we can take a Transformer architecture and replace nn.Linear layers with KANs. We use [minGPT](https://github.com/karpathy/minGPT) as a basis and swap the layers. We then train on a sample corpus and evaluate.

Weights for the configuration in `train.ipynb` can be downloaded here: [link](https://drive.google.com/file/d/12KwPj10H6syJF_I9rvBFzsaa7PnhXhRE/view?usp=share_link).

NOTES:
- I used a [stanford philosophy dataset](https://huggingface.co/datasets/AiresPucrs/stanford-encyclopedia-philosophy) as I struggled to find good english text datasets that could easily be loaded
- I trained the model on a single L4 GPU with high ram in Google Colab
    - Due to this computing constraint, I had to train a tiny version of the model

FUTURE WORK:
- Improve dataset
    - Find a better dataset (more comprehensive)
    - Figure out a better way to load datasets (preferrably only one batch in RAM at a time) to minimize RAM usage
- Increase compute
    - Train model on larger scales to find better performance
    - Evaluate larger models on benchmarks
    - Observe scaling laws, training times, loss patterns, emergent capabilities