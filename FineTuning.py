import torch
import logging
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, TrainerCallback
import torch.nn.functional as F
import multiprocessing
import os
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_datasets(train_dataset, test_dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
    logger.info(f"Datasets saved to {save_dir}")


def load_datasets(save_dir):
    train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pt'))
    test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pt'))
    logger.info(f"Datasets loaded from {save_dir}")
    return train_dataset, test_dataset


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_memory_usage(stage):
    logger.info(f"{stage} - Memory usage: {torch.cuda.memory_allocated() / (1024 * 1024)} MB")


def tokenize_chunk_with_logging(chunk, tokenizer, idx, block_size):
    if idx % 500 == 0:
        logger.info(f"Tokenizing chunk {idx}: {chunk[:50]}...")
    return tokenizer(chunk, return_tensors='pt', padding='max_length', max_length=block_size, truncation=True)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, eos_token, input_block_size=512, output_block_size=256):
        self.examples = []
        self.input_block_size = input_block_size
        self.output_block_size = output_block_size
        self.tokenizer = tokenizer
        self.eos_token = eos_token

        chunk_counter = 0

        for text_index, text in enumerate(texts):
            if text_index % 1000 == 0:
                logger.info(f"Processing text {text_index + 1}/{len(texts)}")
                log_memory_usage("Before processing text")
                gc.collect()

            sentences = text.split('.')
            input_text = ""
            label_text = ""
            current_position = 0

            while current_position < len(sentences):
                while len(input_text) < self.input_block_size and current_position < len(sentences):
                    input_text += sentences[current_position] + "."
                    current_position += 1

                last_dot_pos = input_text.rfind('.')
                if last_dot_pos != -1:
                    input_text = input_text[:last_dot_pos + 1] + self.eos_token

                if current_position < len(sentences):
                    label_text = sentences[current_position]
                    current_position += 1

                    while len(label_text) < self.output_block_size and current_position < len(sentences):
                        label_text += sentences[current_position] + "."
                        current_position += 1

                    last_dot_pos = label_text.rfind('.')
                    if last_dot_pos != -1:
                        label_text = label_text[:last_dot_pos + 1] + self.eos_token

                self.process_chunk(input_text.strip(), label_text.strip(), chunk_counter)
                input_text = label_text
                label_text = ""
                chunk_counter += 1

            # logger.info(f"Processed {chunk_counter} chunks so far")

        logger.info(f"Dataset created with {len(self.examples)} examples")
        log_memory_usage("After dataset creation")
        gc.collect()

    def process_chunk(self, chunk, label, chunk_counter):
        try:
            # logger.info(f"Processing chunk {chunk_counter}: {chunk}")

            tokenized_chunk = tokenize_chunk_with_logging(chunk, self.tokenizer, chunk_counter, self.input_block_size)

            input_ids = tokenized_chunk['input_ids'].squeeze()
            attention_mask = tokenized_chunk['attention_mask'].squeeze()

            # logger.info(f"Label before tokenization for chunk {chunk_counter}: {label}")

            tokenized_labels = self.tokenizer(label, return_tensors='pt', padding='max_length',
                                              max_length=self.output_block_size, truncation=True)
            labels = tokenized_labels['input_ids'].squeeze()

            if len(labels) < self.output_block_size:
                labels = torch.nn.functional.pad(labels, (0, self.output_block_size - len(labels)),
                                                 value=self.tokenizer.pad_token_id)

            self.examples.append({
                'input_ids': input_ids[:self.input_block_size],
                'labels': labels[:self.output_block_size],
                'attention_mask': attention_mask[:self.input_block_size]
            })

            if chunk_counter % 500 == 0:
                logger.info(f"Processed {chunk_counter} chunks")
                log_memory_usage(f"After processing {chunk_counter} chunks")
                gc.collect()

        except Exception as e:
            logger.error(f"Error during tokenization: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100, eos_token_id=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)

    if eos_token_id is not None:
        eos_mask = (target == eos_token_id).squeeze(-1)

        smooth_loss = -lprobs.mean(dim=-1)

        eps_i = epsilon / lprobs.size(-1)

        nll_loss[eos_mask] = (1.0 - epsilon) * nll_loss[eos_mask] + eps_i * smooth_loss[eos_mask]

    if ignore_index is not None:
        valid_mask = target != ignore_index
        nll_loss = nll_loss[valid_mask]

    loss = nll_loss.sum()
    return loss


class CustomTrainer(Trainer):
    def __init__(self, model, args, tokenizer, *other_args, **kwargs):
        super().__init__(model, args, *other_args, **kwargs)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logger.info("Executing compute_loss")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Sample logits: {logits[0, :5, :5]}")
        logger.info(f"Sample labels: {labels[0, :5]}")

        lprobs = F.log_softmax(logits, dim=-1)
        loss = label_smoothed_nll_loss(lprobs, labels, epsilon=0.4, eos_token_id=self.eos_token_id)
        return (loss, outputs) if return_outputs else loss


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    logger.info("Starting script")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    logger.info(f"Local rank: {local_rank}, World size: {world_size}, Rank: {rank}")

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    logger.info(f"Loading tokenizer and model for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'
    model.to(device)

    print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'datasets/m')
    corpus_path = os.path.join(script_dir, 'corpus_total.txt')
    log_memory_usage("before dataset creation")
    print(f"Loading corpus from {corpus_path}")

    def load_corpus_and_split(corpus_path, tokenizer):
        try:
            with open(corpus_path, 'r', encoding='utf-8') as file:
                combined_text = file.read().strip()
                print(f"Total length of corpus: {len(combined_text)} characters")
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return None, None

        split_point = int(len(combined_text) * 0.8)
        train_text = combined_text[:split_point]
        test_text = combined_text[split_point:]

        train_dataset = TextDataset([train_text], tokenizer, eos_token=tokenizer.eos_token)
        test_dataset = TextDataset([test_text], tokenizer, eos_token=tokenizer.eos_token)

        return train_dataset, test_dataset

    train_dataset, test_dataset = load_corpus_and_split(corpus_path, tokenizer)
    if not train_dataset or not test_dataset:
        return
    save_datasets(train_dataset, test_dataset, dataset_dir)

    num_layers_to_freeze = 24
    for i in range(num_layers_to_freeze):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)

    learning_rate = 5e-5
    deepspeed_config = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True
        },
        "gradient_checkpointing": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1
        }
    }

    training_args = TrainingArguments(
        output_dir=os.path.join(script_dir, 'outputs/hf/Modell_Eins'),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=50,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(script_dir, 'logs'),
        logging_steps=10,
        save_total_limit=20,
        load_best_model_at_end=True,
        deepspeed=deepspeed_config if world_size > 1 else None,
        report_to="none",
        learning_rate=learning_rate,
        lr_scheduler_type="reduce_lr_on_plateau",
        weight_decay=0.01
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training error: {e}")

    logger.info("Training complete")

    if local_rank == 0:
        logger.info("Saving model")
        try:
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info(f'Model and tokenizer saved to {training_args.output_dir}')
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    logger.info("Starting evaluation")
    try:
        eval_results = trainer.evaluate()
        logger.info(f"Average test loss: {eval_results['eval_loss']}")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")


if __name__ == "__main__":
    main()



