import os
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
from model import Transformer, TransformerConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TranslationCollator
from datasets import load_from_disk
from tokenizer import FrenchTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##################################
# 1. Global Training Parameters
##################################

CONFIG = {
    "encoder_depth": 6,
    "decoder_depth": 6,
    "mlp_ratio": 4,
    "attention_dropout_p": 0.1,
    "hidden_dropout_p": 0.1,
    "embedding_dimension": 512,
    "num_attention_heads": 8,
    "max_src_len": 512,
    "max_tgt_len": 512,
    "learn_pos_embed": False,

    "path_to_data": "/Users/jhanvi/Desktop/DeepLearning/Project/tokenized_prepare_data",
    "tgt_tokenizer_path": "/Users/jhanvi/Desktop/DeepLearning/Project/trained_tokenizer/french_wp.json",
    "src_tokenizer_model": "google-bert/bert-base-uncased",

    "batch_size": 128,
    "gradient_accumulation_steps": 2,
    "num_workers": 4,

    "learning_rate": 1e-4,
    "training_steps": 150000,
    "warmup_steps": 2000,
    "scheduler_type": "cosine",
    "evaluation_steps": 2500,
    "weight_decay": 0.001,
    "betas": (0.9, 0.98),
    "adam_eps": 1e-6,

    "working_directory": "work_dir",
    "experiment_name": "Seq2Seq_Neural_Machine_Translation",
    "logging_interval": 1,
    "resume_from_checkpoint": None,
}


##################################
# 2. Helper Functions
##################################

def get_tokenizers():
    tgt_tokenizer = FrenchTokenizer(CONFIG["tgt_tokenizer_path"])
    src_tokenizer = AutoTokenizer.from_pretrained(CONFIG["src_tokenizer_model"])
    return src_tokenizer, tgt_tokenizer

def get_dataloaders(src_tokenizer, tgt_tokenizer):
    dataset = load_from_disk(CONFIG["path_to_data"])
    collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
    minibatch_size = CONFIG["batch_size"] // CONFIG["gradient_accumulation_steps"]

    train_loader = DataLoader(dataset["train"],
                              batch_size=minibatch_size,
                              shuffle=True,
                              num_workers=CONFIG["num_workers"],
                              collate_fn=collate_fn)

    test_loader = DataLoader(dataset["test"],
                             batch_size=minibatch_size,
                             shuffle=False,
                             num_workers=CONFIG["num_workers"],
                             collate_fn=collate_fn)
    return dataset, train_loader, test_loader

def build_model(src_tokenizer, tgt_tokenizer):
    config = TransformerConfig(
        embedding_dimension=CONFIG["embedding_dimension"],
        num_attention_heads=CONFIG["num_attention_heads"],
        attention_dropout_p=CONFIG["attention_dropout_p"],
        hidden_dropout_p=CONFIG["hidden_dropout_p"],
        mlp_ratio=CONFIG["mlp_ratio"],
        encoder_depth=CONFIG["encoder_depth"],
        decoder_depth=CONFIG["decoder_depth"],
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        max_src_length=CONFIG["max_src_len"],
        max_tgt_length=CONFIG["max_tgt_len"],
        learn_pos_embed=CONFIG["learn_pos_embed"]
    )
    return Transformer(config)

def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(),
                             lr=CONFIG["learning_rate"],
                             betas=CONFIG["betas"],
                             eps=CONFIG["adam_eps"])

def get_scheduler_fn(optimizer):
    return get_scheduler(
        name=CONFIG["scheduler_type"],
        num_warmup_steps=CONFIG["warmup_steps"],
        optimizer=optimizer,
        num_training_steps=CONFIG["training_steps"]
    )

def sample_input(src_tokenizer):
    return torch.tensor(src_tokenizer("I want to learn something new")['input_ids']).unsqueeze(0)


##################################
# 3. Main Training Entry Point
##################################

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    accelerator = Accelerator(
        project_dir=os.path.join(CONFIG["working_directory"], CONFIG["experiment_name"]),
        log_with="wandb"
    )

    # Setup components
    src_tokenizer, tgt_tokenizer = get_tokenizers()
    dataset, train_loader, test_loader = get_dataloaders(src_tokenizer, tgt_tokenizer)
    model = build_model(src_tokenizer, tgt_tokenizer)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler_fn(optimizer)
    loss_fn = nn.CrossEntropyLoss()
    src_ids = sample_input(src_tokenizer)

    # Prepare with accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    accelerator.register_for_checkpointing(scheduler)
    accelerator.print(dataset)
    accelerator.print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Resume checkpoint
    completed_steps = 0
    if CONFIG["resume_from_checkpoint"] is not None:
        path = os.path.join(CONFIG["working_directory"], CONFIG["experiment_name"], CONFIG["resume_from_checkpoint"])
        with accelerator.main_process_first():
            accelerator.load_state(path)
        completed_steps = int(CONFIG["resume_from_checkpoint"].split("_")[-1])
        accelerator.print(f"Resuming from iteration: {completed_steps}")

    train = True
    progress_bar = tqdm(range(completed_steps, CONFIG["training_steps"]), disable=not accelerator.is_main_process)

    while train:
        accumulate_step = 0
        accumulate_loss = 0
        accuracy = 0

        for batch in train_loader:
            src_input_ids = batch['src_input_ids'].to(accelerator.device)
            src_pad_mask = batch['src_pad_mask'].to(accelerator.device)
            tgt_input_ids = batch['tgt_input_ids'].to(accelerator.device)
            tgt_pad_mask = batch['tgt_pad_mask'].to(accelerator.device)
            tgt_outputs = batch['tgt_outputs'].to(accelerator.device)

            output = model(src_input_ids, tgt_input_ids, src_pad_mask, tgt_pad_mask)
            output = output.flatten(0, 1)
            tgt_outputs = tgt_outputs.flatten()

            loss = loss_fn(output, tgt_outputs) / CONFIG["gradient_accumulation_steps"]
            accumulate_loss += loss
            accelerator.backward(loss)

            pred = output.argmax(dim=-1)
            mask = tgt_outputs != -100
            acc = (pred[mask] == tgt_outputs[mask]).sum() / len(pred[mask])
            accuracy += acc / CONFIG["gradient_accumulation_steps"]

            accumulate_step += 1

            if accumulate_step % CONFIG["gradient_accumulation_steps"] == 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if accumulate_step % CONFIG["logging_interval"] == 0:
                    accumulate_loss = accumulate_loss.detach()
                    accuracy = accuracy.detach()
                    if accelerator.num_processes > 1:
                        accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                    accelerator.log({
                        "train_loss": accumulate_loss,
                        "accuracy": accuracy,
                        "learning_rate": scheduler.get_last_lr()[0]
                    }, step=completed_steps)

                    if accelerator.is_main_process:
                        progress_bar.write(f"[{completed_steps}/{CONFIG['training_steps']}] Loss: {accumulate_loss} | Acc: {accuracy}")

                if completed_steps % CONFIG["evaluation_steps"] == 0:
                    model.eval()
                    test_losses, test_accs = [], []

                    for batch in tqdm(test_loader, disable=not accelerator.is_main_process):
                        with torch.inference_mode():
                            output = model(batch['src_input_ids'].to(accelerator.device),
                                           batch['tgt_input_ids'].to(accelerator.device),
                                           batch['src_pad_mask'].to(accelerator.device),
                                           batch['tgt_pad_mask'].to(accelerator.device))
                        output = output.flatten(0, 1)
                        tgt_outputs = batch['tgt_outputs'].to(accelerator.device).flatten()

                        loss = loss_fn(output, tgt_outputs)
                        pred = output.argmax(dim=-1)
                        mask = tgt_outputs != -100
                        acc = (pred[mask] == tgt_outputs[mask]).sum() / len(pred[mask])

                        if accelerator.num_processes > 1:
                            loss = torch.mean(accelerator.gather_for_metrics(loss.detach()))
                            acc = torch.mean(accelerator.gather_for_metrics(acc.detach()))

                        test_losses.append(loss.item())
                        test_accs.append(acc.item())

                    test_loss = np.mean(test_losses)
                    test_acc = np.mean(test_accs)
                    accelerator.log({"test_loss": test_loss, "test_acc": test_acc}, step=completed_steps)

                    if accelerator.is_main_process:
                        progress_bar.write(f"Testing Loss: {test_loss} | Test Acc: {test_acc}")

                    accelerator.save_state(os.path.join(CONFIG["working_directory"], CONFIG["experiment_name"], f"checkpoint_{completed_steps}"))

                    if accelerator.is_main_process:
                        translated = accelerator.unwrap_model(model).inference(
                            src_ids.to(accelerator.device),
                            tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
                            tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"]
                        )
                        translated = tgt_tokenizer.decode(translated, skip_special_tokens=False)
                        accelerator.print(f"Translation: {translated}")

                    model.train()

                completed_steps += 1
                progress_bar.update(1)
                accumulate_loss = 0
                accuracy = 0

                if completed_steps >= CONFIG["training_steps"]:
                    train = False
                    accelerator.save_state(os.path.join(CONFIG["working_directory"], CONFIG["experiment_name"], "final_checkpoint"))
                    break
