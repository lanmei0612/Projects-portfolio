from setfit import AbsaModel
from datasets import load_dataset
from setfit import AbsaTrainer, TrainingArguments
from transformers import EarlyStoppingCallback


model = AbsaModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    spacy_model="en_core_web_sm",
)


# The training/eval dataset must have `text`, `span`, `label`, and `ordinal` columns
dataset = load_dataset("tomaarsen/setfit-absa-semeval-restaurants", split="train")
train_dataset = dataset.select(range(128))
eval_dataset = dataset.select(range(128, 256))


args = TrainingArguments(
    output_dir="models",
    num_epochs=5,
    use_amp=True,
    batch_size=300,
    eval_strategy="steps",
    eval_steps=15,
    save_steps=15,
    load_best_model_at_end=True,
    )


trainer = AbsaTrainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )


trainer.train()


metrics = trainer.evaluate(eval_dataset)
print(metrics)


model.save_pretrained("models/setfit-absa-model")
