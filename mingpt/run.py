import os
import time
import math

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, setup_logging, CfgNode as CN
from trainer import Trainer
from model import GPT

from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------

class WikitextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every token to an integer
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_wikitext(block_size):
    # Load the WikiText-103 dataset using Hugging Face datasets
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    
    # Tokenize the dataset
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    def tokenize(examples):
        return tokenizer(examples['text'])
    dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=['text'])
    
    # Flatten the dataset
    dataset = dataset.flatten()
    
    # Convert to tensors
    dataset = dataset.with_format('torch')
    
    # Create the WikitextDataset
    train_dataset = WikitextDataset(dataset['input_ids'], block_size)
    return train_dataset, tokenizer

# -----------------------------------------------------------------------------

def run_training(trainer, total_iters):
    # Initialize tqdm progress bar
    pbar = tqdm(total=total_iters, desc="Training progress", leave=True)

    def update_progress(trainer):
        """Callback function to update progress and display loss after each iteration."""
        # Update the progress bar by one step
        pbar.update(1)
        # Set the loss in the postfix of the tqdm bar to display current loss
        pbar.set_postfix_str(f"Loss: {trainer.loss.item():.4f}")

    # Attach the progress updating callback
    trainer.add_callback('on_batch_end', update_progress)

    try:
        # Start the training
        trainer.run()
    finally:
        # Ensure the progress bar closes correctly
        pbar.close()

def main():
    # Prepare your trainer as usual
    set_seed(42)
    C = CN()
    C.system = CN()
    C.system.work_dir = './out/wikitext'
    setup_logging(C)

    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'
    C.model.vocab_size = 50257
    C.model.block_size = 128

    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.max_iters = 20000
    C.trainer.batch_size = 256

    model = GPT(C.model)
    train_dataset, tokenizer = load_wikitext(C.model.block_size)

    trainer = Trainer(C.trainer, model, train_dataset)

    # Run the training with tqdm integration
    run_training(trainer, C.trainer.max_iters)

    # Remaining parts of your script
    model_path = os.path.join(C.system.work_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    loaded_model = GPT(C.model)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    input_phrase = "The quick brown fox"
    input_ids = tokenizer.encode(input_phrase, return_tensors='pt')
    with torch.no_grad():
        output = loaded_model.generate(input_ids, max_new_tokens=50)
    generated_text = tokenizer.decode(output[0])
    print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main()