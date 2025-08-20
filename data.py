import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tokenizer import FrenchTokenizer
import pandas as pd
from tqdm import tqdm
##from model import TransformerConfig

# def TranslationCollator(src_tokenizer, tgt_tokenizer):
    # """
    # Collate Function of Seq2Seq Translation:

    # Returns:
    #     src_input_ids: Batch of tokenized english padded with its pad token (src_pad_token)
    #     src_pad_mask: Which tokens in our tensor are padding (so we can ignore them in attention)
    #     tgt_input_ids: Given all our tgt_ids, we set up a causal tgt input by taking all but the last index
    #     tgt_pad_mask: Which tokens in our tgt_input_ids are padding tokens (tgt_pad_token)
    #     tgt_outputs: Next token prediction of the tgt_input_ids
    # """
#     def _collate_fn(batch):
        
#         src_ids = [torch.tensor(i["src_ids"]) for i in batch]
#         tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]
        
#         src_pad_token = src_tokenizer.pad_token_id
#         src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
#         src_pad_mask = (src_padded != src_pad_token)
        
#         tgt_pad_token = tgt_tokenizer.special_tokens_dict['[PAD]']
#         tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)
#         tgt_pad_mask = (tgt_padded != tgt_pad_token)
        
#         input_tgt = tgt_padded[:, :-1].clone()# Exclude the last token for input ----[w1, w2, w3, PAD] -> input - [w1, w2, w3]
#         output_tgt = tgt_padded[:, 1:].clone()# Exclude the first token for output --[w1, w2, w3, PAD] -> output - [w2, w3, PAD]
        
        
#         ## Why we use the input_tgt_mask and Output_tgt_mask?
#         ## input_tgt_mask is used to mask the padding tokens in the input target sequence Input - [w1, w2, w3, PAD] -> output - [w2, w3, PAD]
#         ## output_tgt_mask is used to mask the padding tokens in the output target sequence Output - [w2, w3, PAD] -> output - [w3, PAD]
#         ## In the Cross Entropy Loss, we will use the -100 for the padding tokens so that they are not considered in the loss calculation [PAD] -> [-100]
#         input_tgt_mask = (input_tgt != tgt_pad_token)
#         output_tgt[output_tgt == tgt_pad_token] = -100
        
#         batch = { "src_input_ids": src_padded, ## Input source sequence excluding the last token
#                  "src_pad_mask": src_pad_mask, ## Mask for the input source sequence
#                  "tgt_input_ids": input_tgt, ##  Output- Input target sequence excluding the last token
#                  "tgt_pad_mask": input_tgt_mask, ## Mask for the input target sequence
#                  "tgt_outputs": output_tgt} ## Next token output prediction
        
        
#         return batch
#         #print(f"Source Padded: {src_padded}")
#         #print(f"Source Padded Shape: {src_padded.shape}")
#         #print(f"Target Padded: {tgt_padded}")
#         #print(f"Target Padded Shape: {tgt_padded.shape}")
#         #print(f"Source Pad Mask: {src_pad_mask}")
#         #print(f"Target Pad Mask: {tgt_pad_mask}")
        
#     return _collate_fn

class TranslationCollator:
    
    """
    Collate Function of Seq2Seq Translation:

    Returns:
        src_input_ids: Batch of tokenized english padded with its pad token (src_pad_token)
        src_pad_mask: Which tokens in our tensor are padding (so we can ignore them in attention)
        tgt_input_ids: Given all our tgt_ids, we set up a causal tgt input by taking all but the last index
        tgt_pad_mask: Which tokens in our tgt_input_ids are padding tokens (tgt_pad_token)
        tgt_outputs: Next token prediction of the tgt_input_ids
    """
    
    def __init__(self, src_tokenizer, tgt_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __call__(self, batch):
        src_ids = [torch.tensor(i["src_ids"]) for i in batch]
        tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]

        src_pad_token = self.src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
        src_pad_mask = (src_padded != src_pad_token)

        tgt_pad_token = self.tgt_tokenizer.special_tokens_dict['[PAD]']
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)
        tgt_pad_mask = (tgt_padded != tgt_pad_token)

        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()

        input_tgt_mask = (input_tgt != tgt_pad_token)
        output_tgt[output_tgt == tgt_pad_token] = -100

        batch = {
            "src_input_ids": src_padded,
            "src_pad_mask": src_pad_mask,
            "tgt_input_ids": input_tgt,
            "tgt_pad_mask": input_tgt_mask,
            "tgt_outputs": output_tgt,
        }
        return batch

        
if __name__ == '__main__':
    
    path_to_data_root = "/Users/jhanvi/Desktop/DeepLearning/Project/Dataset"
    path_to_save_raw = "/Users/jhanvi/Desktop/DeepLearning/Project/raw_data"
    path_to_save_tok = "/Users/jhanvi/Desktop/DeepLearning/Project/tokenized_prepare_data"
    
    dataset = load_from_disk(path_to_save_tok)
    
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tgt_tokenizer = FrenchTokenizer("/Users/jhanvi/Desktop/DeepLearning/Project/trained_tokenizer/french_wp.json")
    ##print(dataset['train'][0])
    collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
    
    loader = DataLoader(dataset['train'],
                        batch_size=128,
                        collate_fn=collate_fn,
                        shuffle=True,
                        num_workers=32)
    from tqdm import tqdm
    for samples in tqdm(loader):
            pass
    
    
## Error occur becz the collate function is not implemented correctly like the each element in the batch should be of equal size which is now preformed in the collate function
#    python data.py
# Traceback (most recent call last):
#   File "/Users/jhanvi/Desktop/DeepLearning/Project/data.py", line 36, in <module>
#     for samples in loader:
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 733, in __next__
#     data = self._next_data()
#            ^^^^^^^^^^^^^^^^^
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 789, in _next_data
#     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
#     return self.collate_fn(data)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
#     return collate(batch, collate_fn_map=default_collate_fn_map)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 172, in collate
#     key: collate(
#          ^^^^^^^^
#   File "/Users/jhanvi/Desktop/DeepLearning/venv/lib/python3.12/site-packages/torch/utils/data/_utils/collate.py", line 207, in collate
#     raise RuntimeError("each element in list of batch should be of equal size")
# RuntimeError: each element in list of batch should be of equal size