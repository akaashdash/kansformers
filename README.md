# Kansformers: Transformers using KANs

[Kolmogorov-Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) are a proposed alternative architecture to MLPs. Despite having similar capabilities (and being representable with MLPs), KANs allow for faster training, less forgetting, and more interpretability.

With some modifications, KANs can be switched out for pytorch nn.Linear layers. We first use [efficient KANs](https://github.com/Blealtan/efficient-kan) and [change shape processing](https://github.com/Blealtan/efficient-kan/pull/6) to allow for better compatability. 

Because the two are interchangable, we can take a Transformer architecture and replace nn.Linear layers with KANs. We use [minGPT](https://github.com/karpathy/minGPT) as a basis and swap the layers. We then train on a sample corpus and evaluate.

## Running the model

The `train.ipynb` demonstrates a sample run of the model. Any further explanation should be found on the[ minGPT repository](https://github.com/karpathy/minGPT).

## Checkpoints

Weights for several checkpoints can be found here: [link](https://drive.google.com/drive/folders/1qYOhLGMI3MGbzZhRF8rXk47KqhrURq19?usp=share_link)

The `model-5-5-2024.pt` checkpoint uses: `n_layer=2, n_head=2, n_embd=128`
- This model is trained on [Stanford philosophy](https://huggingface.co/datasets/AiresPucrs/stanford-encyclopedia-philosophy)

The `model-5-6-2024.pt` checkpoint uses: `model_type = 'gpt-micro'`
- This model is trained on [tinyshakespeare](https://github.com/karpathy/llm.c/blob/master/prepro_tinyshakespeare.py)

## Notes
- I trained the model on a single L4 GPU with high ram in Google Colab
    - Due to this computing constraint, I had to train a tiny version of the model
- Efficient KAN is used as it is currently the strongest implementation of KAN: [benchmarks](https://github.com/GistNoesis/FusedFourierKAN/issues/4)
    - I had initially planned to use some c/c++ implementation of KAN to improve times but benchmarks show that current implementation is acceptable
    - I am not sure if there is any benchmark of the model memory footprint (not forward/backward pass memory) across the implementations, but I assume efficient KAN will still be the best

## Future Work
- Improve dataset
    - Find a better dataset (more comprehensive)
        - Preferrably GPT-2 dataset
    - Figure out a better way to load datasets (preferrably only one batch in RAM at a time) to minimize RAM usage
- Increase compute
    - Train model on larger scales to find better performance
        - Preferrable GPT-2 scale
    - Evaluate larger models on benchmarks
    - Observe scaling laws, training times, loss patterns, emergent capabilities
- Look at other transformer models (BERT, BART, DistilBERT, RoBERTa)
    - With modified efficient kan, should be easy to swap out nn.Linear layers with KAN layers
        - Might be able to find a systematic way to do this