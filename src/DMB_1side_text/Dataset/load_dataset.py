"""
Based on the code of https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""
from datasets import load_dataset, load_from_disk
import requests
import json
from itertools import chain
from .detokenize import (lm1b_detokenizer, ptb_detokenizer, 
                         wt_detokenizer, lambada_detokenizer)
from torch.utils.data import DataLoader, DistributedSampler
from .dataset_text8 import Text8Dataset
from datasets import Dataset

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(0)
        for data in dataloader:
            yield data

def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset



def get_dataset(name, 
                tokenizer=None,
                mode="train", 
                cache_dir=None, 
                seqlen=1024, 
                num_proc=8,
                debug=False):
    """Dataset.

    Load tokenized dataset
        1) wikitext103
        2) wikitext2
        3) ptb
        4) lambada
        5) text8: specified tokenizer 
                    (written in dataset_text8.py)

    Args: 
        name: eg. "wikitext103", "ptb",...
        tokenizer
        mode: train, valid, test
        cache_dir: cache directory
        seqlen
        num_proc: number of processes for tokenization

    Returns:
        dataset: tokenized and chunked dataset
    """
    if name == "wikitext103":
#        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
        dataset = load_from_disk(f"./cache/tokenized_wikitext103_{mode}")
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif name == "openwebtext":
#        dataset = load_from_disk(f"./cache/tokenized_openwebtext_{mode}")
        dataset = load_dataset(name, 
                               cache_dir=cache_dir, 
                               trust_remote_code=True)
    elif name=="text8":
        # torch dataset
        # tokenizer is not needed for text8
        dataset = Text8Dataset(root=cache_dir, 
                               seq_len=seqlen, 
                               split=mode, 
                               debug=debug,
                               download=True)
    else:
        dataset = load_dataset(name, 
                               cache_dir=cache_dir, 
                               trust_remote_code=True)

    if name == "lambada" or name=="text8":
        dataset = dataset
    else:
        dataset = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    if tokenizer is not None:
        EOS = tokenizer.encode(tokenizer.eos_token)[0]
        def preprocess_and_tokenize(example):
            if name == "ptb":
                text = example['sentence']
            else:
                text = example["text"]
            
            if detokenizer is not None:
                text = _apply_detokenizer(detokenizer)(text)

            tokens = tokenizer(text, return_attention_mask=False)
            # add in EOS token following 
            # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
            for token in tokens['input_ids']:
                token.append(EOS)
            return tokens

        dataset = dataset.map(preprocess_and_tokenize, 
                              batched=True, 
                              num_proc=num_proc, 
                              load_from_cache_file=True)

        if name == "ptb":
            dataset = dataset.remove_columns('sentence')
        else:
            dataset = dataset.remove_columns('text')

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) 
                                     for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < seqlen  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // seqlen) * seqlen
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + seqlen] for i in range(0, total_length, seqlen)]
                for k, t in concatenated_examples.items()
            }
            return result

        dataset = dataset.map(group_texts, 
                              batched=True, 
                              num_proc=num_proc, 
                              load_from_cache_file=True)
        dataset = dataset.with_format('torch')
    return dataset


def get_dataloaders(name, 
                    tokenizer, 
                    batch_size, 
                    accumulation_step, 
                    cache_dir, 
                    seqlen, 
                    shuffle, 
                    ngpus=1, 
                    distributed=True,
                    mode="train", 
                    debug=False):
    """DataLoader.

    Return the dataloaders for training

    Args:
        name: dataset name
        tokenizer
        batch_size
        accumulation_step: gradient accumulation step num
        cache_dir
        seqlen: block size for chunking
        shuffle [bool]
        ngpus: number of gpus
        distributed: whether to use distributed training
        mode: train or valid
        debug: debug mode

    Returns:
        dataloader
        tokenized_dataset

    Raises:
        ValueError: 1) Train: batch_size % (ngpus * accumulation_step) != 0
                    2) Validation and Test: batch_size % ngpus != 0
    """

    tokenized_dataset = get_dataset(name=name,
                                    tokenizer=tokenizer,
                                    mode=mode, 
                                    cache_dir=cache_dir, 
                                    seqlen=seqlen,
                                    debug=debug)
    if name == "openwebtext":
        tokenized_dataset.save_to_disk("./cache/tokenized_openwebtext_train")

    if debug and name != "text8":
        tokenized_dataset = tokenized_dataset.select(range(32))

    sampler = None
    if distributed:
        sampler = DistributedSampler(tokenized_dataset)
        shuffle = False  # Shuffling is handled by the DistributedSampler

    if mode=="train":
        if batch_size % (ngpus * accumulation_step) != 0:
            raise ValueError((f"Train Batch Size {batch_size} "
                              f"is not divisible by {ngpus} gpus "
                              f"with accumulation {accumulation_step}."))
        data_bs = batch_size // (ngpus * accumulation_step)
    elif mode=="validation":
        if batch_size % ngpus != 0:
            raise ValueError((f"Validation Batch Size {batch_size} "
                              f"is not divisible by {ngpus} gpus."))
        data_bs = batch_size // ngpus
    elif mode=="test":
        if batch_size % ngpus != 0:
            raise ValueError((f"Test Batch Size {batch_size} "
                              f"is not divisible by {ngpus} gpus."))
        data_bs = batch_size // ngpus
    else:
        raise ValueError(f"Invalid mode: {mode}")


    dataloader = DataLoader(tokenized_dataset, 
                            batch_size=data_bs,
                            shuffle=shuffle if not distributed else False,
                            sampler=sampler)
    return dataloader, tokenized_dataset
