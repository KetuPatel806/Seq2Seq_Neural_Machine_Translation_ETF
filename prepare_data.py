import os
import argparse
from datasets import load_dataset,concatenate_datasets,load_from_disk
from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer

def build_english2french_dataset(path_to_data_root,path_to_save_raw,test_prop=0.005,cache_dir=None):
    
    """
    This processes en-fr data found in https://www.statmt.org/wmt14/translation-task.html
    I just downloaded the data and created a folder structure like:

    └── english2french/
        ├── common_crawl/
        │   ├── commoncrawl.fr-en.en
        │   └── commoncrawl.fr-en.fr
        ├── europarl/
        │   ├── europarl-v7.fr-en.en
        │   └── europarl-v7.fr-en.fr
        ├── giga_french/
        │   ├── giga-fren.release2.fixed.en
        │   └── giga-fren.release2.fixed.fr
        └── un_corpus/
            ├── undoc.2000.fr-en.en
            └── undoc.2000.fr-en.ft

    This provides about 15GB of data for us to train on!

    This function will take all these datasets and merge them into a single
    Huggingface Dataset!
    
    """
    hf_datasets = []
    ## List all the directories in the path_to_data_root
    for dir in os.listdir(path_to_data_root):
        
        ## Skip if not a directory
        path_to_dir = os.path.join(path_to_data_root,dir)
        
        if os.path.isdir(path_to_dir):
            
            print("Processing directory: ",path_to_dir)
            
            french_text = english_text = None
            ## Find the french and english text files in the directory
            ## We assume the files are named with .fr and .en extensions respectively
            ## If the files are named differently, we can modify this logic accordingly
            for txt in os.listdir(path_to_dir):
                
                if txt.endswith(".fr"):
                    french_text = os.path.join(path_to_dir,txt)
                elif txt.endswith(".en"):
                    english_text = os.path.join(path_to_dir,txt)
            ## If both french and english text files are found, load them into a Huggingface Dataset
            ## and add them to the list of datasets
            ## If not, skip this directory
            ## Note: We assume that the french text file has the same name as the english text       
            if french_text is not None and english_text is not None:
                
                french_dataset = load_dataset("text",data_files=french_text,cache_dir=cache_dir)["train"]
                english_dataset = load_dataset("text",data_files=english_text,cache_dir=cache_dir)["train"]
                ## Here we create the dataset with the columns so the tebular data is in the format we want
                ## We rename the columns to "text" and "label" for the Huggingface
                english_dataset = english_dataset.rename_column("text","english")
                english_dataset = english_dataset.add_column("french",french_dataset["text"])
                ## Append the dataset to the list of datasets
                hf_datasets.append(english_dataset)
                
    hf_datasets = concatenate_datasets(hf_datasets)
    print(hf_datasets)
    ## This is not the train test split of the dataset, but rather the split of the entire dataset into train and test
    hf_datasets = hf_datasets.train_test_split(test_size=test_prop)
    
    ## Save the raw dataset to disk
    hf_datasets.save_to_disk(path_to_save_raw)
    
def tokenize_english2french_dataset(path_to_save_raw,
                                    path_to_save_tok,
                                    num_workers=24,
                                    max_length=512,
                                    min_length=5,
                                    truncate=False):
    """
    It is easier to pre-tokenize our data before training rather than tokenizing on the fly, so we can 
    set that up here! We will be using the FrenchTokenizer that we trained in `tokenizer.py` as well as 
    the regular BERT tokenizer for our english encoder.

    Caveats: 
    
    In the default model setup, we can only do a max sequence length of 512, so we need to make sure
    to truncate anything thats longer. This probably isn't good all the time as you cannot just delete words from
    the english input or french output without messing up the translation, but there are very few cases of this so we 
    will just do it this way.

    We also set a min length of 5 on the targets. Our targets will be a Start Token + End Token + French tokens, so 
    by setting a min lenght of 5, we are saying the actual sentence (not including our special tokens) has atleast 
    3 tokens. This way we actually have some tokens to do the causal modeling (and there may be some blank strings in 
    our data so we can clean this up) 
    
    """
    ## French Tokenizer is a custom tokenizer that we trained on the French dataset Use the WordPiece tokenizer
    ## We will use the AutoTokenizer from Huggingface to load the BERT tokenizer for English
    ## We will use the FrenchTokenizer that we trained in `tokenizer.py` to
    french_tokenizer = FrenchTokenizer("/Users/jhanvi/Desktop/DeepLearning/Project/trained_tokenizer/french_wp.json",truncate=truncate,max_length=max_length)
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    ## Raw dataset is loaded from the path_to_save_raw
    raw_dataset = load_from_disk(path_to_save_raw)
    
    ## We will now tokenize the dataset using the FrenchTokenizer and the BERT tokenizer
    ## We will create a function that takes in the examples and returns the tokenized examples
    def _tokenized_text(examples):
        ## Source will be the English text and target will be the French text
        ## We will use the FrenchTokenizer to tokenize the French text and the BERT tokenizer to english text
        english_text = examples["english"]
        french_text = examples["french"]
        
        src_ids = english_tokenizer(english_text,truncation=True,max_length=max_length)['input_ids']
        tgt_ids = french_tokenizer.encode(french_text)
        
        batch = {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
        ##print(batch)
        return batch
    
    tokenized_dataset = raw_dataset.map(_tokenized_text,batched=True,num_proc=num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english","french"])
    
    filter_func = lambda batch: [len(e) > min_length for e in batch['tgt_ids']]
    tokenized_dataset = tokenized_dataset.filter(filter_func,batched=True)
    
    tokenized_dataset.save_to_disk(path_to_save_tok)
    
    

if __name__ == "__main__":
    
    path_to_data_root = "/Users/jhanvi/Desktop/DeepLearning/Project/Dataset"
    path_to_save_raw = "/Users/jhanvi/Desktop/DeepLearning/Project/raw_data"
    path_to_save_tok = "/Users/jhanvi/Desktop/DeepLearning/Project/tokenized_prepare_data"
    
    ##build_english2french_dataset(path_to_data_root,path_to_save_raw)
    tokenize_english2french_dataset(path_to_save_raw,
                                    path_to_save_tok)
    