# Kansformers: Transformers using KANs

[Kolmogorov-Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) are a proposed alternative architecture to MLPs. Despite having similar capabilities (and being representable with MLPs), KANs allow for faster training, less forgetting, and more interpretability.

With some modifications, KANs can be switched out for pytorch nn.Linear layers. We first use [efficient KANs](https://github.com/Blealtan/efficient-kan) and [change shape processing](https://github.com/Blealtan/efficient-kan/pull/6) to allow for better compatability. 

Because the two are interchangable, we can take a Transformer architecture and replace nn.Linear layers with KANs. We use [minGPT](https://github.com/karpathy/minGPT) as a basis and swap the layers. We then train on a sample corpus and evaluate.

## Running the model

The `train.ipynb` demonstrates a sample run of the model. Any further explanation should be found on the[ minGPT repository](https://github.com/karpathy/minGPT).

## Checkpoints

Weights for several checkpoints can be found here: [link](https://drive.google.com/drive/folders/1qYOhLGMI3MGbzZhRF8rXk47KqhrURq19?usp=share_link)

The `model-5-5-2024.pt` checkpoint uses: `n_layer=2, n_head=2, n_embd=128, C.model.block_size = 128`
- This model is trained on [Stanford philosophy](https://huggingface.co/datasets/AiresPucrs/stanford-encyclopedia-philosophy)

The `model-5-7-2024-1.pt` checkpoint uses: `model_type = 'gpt-micro', C.model.block_size = 128`
- This model is trained on [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Used ~18GB vRAM

The `model-5-7-2024-2.pt` checkpoint uses: `model_type = 'gpt-micro', C.model.block_size = 256`
- This model is trained on [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Used ~37GB vRAM
- Notebook at this commit is wrong :( the actual one didn't save on my drive
    - Weights are okay tho

The `model-5-7-2024-3.pt` checkpoint uses: `model_type = 'gpt-nano', C.model.block_size = 128`
- This model is trained on [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Used ~11GB vRAM

The `model-5-7-2024-4.pt` checkpoint uses: `model_type = 'gpt-nano', C.model.block_size = 256`
- This model is trained on [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Used ~25GB vRAM

The `model-5-7-2024-5.pt` checkpoint uses: `model_type = 'gpt-mini', C.model.block_size = 128`
- This model is trained on [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Used ~30B vRAM

## Observations
### Training RAM
- Doubling the block size (context window) doubled the amount of RAM required to train

### Block/Model Size vs. Loss/Generations
- Both the 128 and 256 block sizes for `gpt-micro` sit at the similar low 6.xxxx loss values when training stops, although 256 does slightly better by reaching high 5.xxxx
    - Could indicate that model size makes more of a difference than block size
    - Outputs were significantly better for 256 block size
- Training a smaller model, `gpt-nano` with block size 128 gives a mid 6.xxxx loss and terrible outputs
    - Training with block size 256 gives the same loss and output quality
    - Could indicate that block size and model size must scale up together to see noticeable difference
- When setting temperature to below 1, text becomes full of special characters; When setting to above 1, text becomes gibberish
    - Could be an issue with model size or temperature is not having desired effects
- For `gpt-mini` with block size 128, minimal testing indicates that a temperature of about 0.75 produces the best outputs

## Notes
- I trained the model on a single L4 GPU with high ram in Google Colab
    - Due to this computing constraint, I had to train a tiny version of the model
- Checkpoints `model-5-7-2024-2.pt`, `model-5-7-2024-4.pt`, and `model-5-7-2024-5.pt` were trained on a single high-ram A100 in Google Colab
- Efficient KAN is used as it is currently the strongest implementation of KAN: [benchmarks](https://github.com/Jerry-Master/KAN-benchmarking)
    - I had initially planned to use some c/c++ implementation of KAN to improve times but benchmarks show that current implementation is acceptable
    - I am not sure if there is any benchmark of the model memory footprint (not forward/backward pass memory) across the implementations, but I assume efficient KAN will still be the best
- Early stopping was used due to its proven effectiveness in LLMs: [paper](https://arxiv.org/abs/2001.08361)
- Randomization is important, the dataset uses sampling and no seed is set

## Future Work
- Increase compute
    - Train model on larger scales to find better performance
        - Preferrable GPT-2 scale: `model_type = 'gpt2', C.model.block_size = 1024`
    - Evaluate larger models on benchmarks
    - Observe scaling laws, training times, loss patterns, emergent capabilities
- Look at other transformer models (BERT, BART, DistilBERT, RoBERTa)
    - With modified efficient kan, should be easy to swap out nn.Linear layers with KAN layers
        - Might be able to find a systematic way to do this
- Maybe control sampling/randomness with a set seed for replication