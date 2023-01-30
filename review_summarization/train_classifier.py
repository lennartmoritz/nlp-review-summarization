import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig, TrainingArguments, Trainer, DistilBertTokenizer, \
    DataCollator, DataCollatorWithPadding
from datasets import load_dataset


def train_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = DistilBertForSequenceClassification(config)
    model.distilbert.from_pretrained('distilbert-base-uncased')
    for param in model.distilbert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    dataset = load_dataset('sst2')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_dataset = dataset.map(lambda d: tokenizer(d['sentence']).to(device), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    train_model()
