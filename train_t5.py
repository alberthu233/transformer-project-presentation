import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class PromptDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        self.original_text = data['original_text']
        self.rewritten_text = data['rewritten_text']
        self.prompt_text = data['rewrite_prompt']
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.original_text)

    def __getitem__(self, idx):
        original_text = self.original_text[idx]
        rewritten_text = self.rewritten_text[idx]
        prompt_text = self.prompt_text[idx]

        task_prefix = f"Recover the original instruction of the following text transformation: original text: {original_text}, transformed text: {rewritten_text}"

        encoding = self.tokenizer(
            task_prefix,
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            prompt_text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids.squeeze(),
            'attention_mask': attention_mask.squeeze(),
            'labels': labels.squeeze(),
        }

class PromptDecoderLightning(pl.LightningModule):
    def __init__(self, model, tokenizer, max_source_length, max_target_length):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_target_length)
        predicted_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predicted_text

    def generate(self, original_text, rewritten_text):
        task_prefix = f"Recover the original instruction of the following text transformation: original text: {original_text}, transformed text: {rewritten_text}"
        encoding = self.tokenizer(
            task_prefix,
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_target_length)
        predicted_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predicted_text

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Define the maximum source and target lengths
max_source_length = 512
max_target_length = 128

# Load and preprocess the dataset
df = pd.read_csv('data/rewrite_text_10k.csv')
# select the first 1000 rows for testing
df_train = df.head(8000)
dataset = PromptDataset(df_train, tokenizer, max_source_length, max_target_length)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model and trainer
prompt_decoder = PromptDecoderLightning(model, tokenizer, max_source_length, max_target_length)

checkpoint_callback = ModelCheckpoint(
    dirpath='ckpt/',
    filename='prompt-t5-sr-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)

logger = TensorBoardLogger(save_dir="logger")
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=100,
    accelerator="gpu",
    devices=2
)

trainer.fit(prompt_decoder, train_loader, val_loader)