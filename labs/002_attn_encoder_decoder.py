# README: try impelementing each and every function using multi-threading for each function

import math
import random
import spacy
import time

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset

from mycustom_coded_decoder import MyCustomDecoder
from mycustom_coded_encoder import MyCustomEncoder
from mycustom_coded_seq2seq import Seq2Seq

SEED = 23
DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backend.cudnn.deterministic = True

# this will load the brain of particular lanaguage, contains all in one pipeline for preprocessing language text.
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# reverse the source tokens, because in research paper we find out, reversing improves the quality of translation.
# this is totally early days strategy, modern days encoder decoder specially those with attention mechanism do not required this.
# before this let's see what's inside the spacy_de or spacy_en.
# It is of instance of spacy.lang.Language (parent class) >> spacy.lang.de.German (actually this one)
print(spacy_de.pipe_names)

for _ in spacy_de.pipe_names:
    print(_, type(_))  # this prints what `_` type is

# this will print list of `str`, these are specialized processing units.
# ['tok2vec', 'tagger', 'morphologizer', 'parser', 'lemmatizer', 'attribute_ruler', 'ner']
# tok2vec: convert token into dense vector
# tagger: assign POS
# morphologizer: analyze word features (case, gender, number)
# parser: Identify grammatical sturcture
# lemmatizer: reduce word to base root form
# ner: Identify name entity recognition

# Now, let just first load the dataset
raw_dataset = load_dataset("bentrevett/multi30k")
print(raw_dataset)

# How a single item looks like in the existing dataset?
print(f"{raw_dataset['train']['en'][0]} -> {raw_dataset['train']['de'][0]}")

# Now moving to next part is tokenization
def sentences_into_tokens(sentences:list, src:bool=False, tokenizer=None):
    if not tokenizer:
        return None
    tokens = [ token.text.lower() for token in tokenizer(sentences) ]
    return tokens if not src else tokens[::-1]

# tokenization function
def tokenize_function(dataset: Dataset):
    return {
        'source': [sentences_into_tokens(sentences, src=True, tokenizer=spacy_de) for sentences in dataset['de']],  # source sentence
        'target': [sentences_into_tokens(sentences, src=False, tokenizer=spacy_en) for sentences in dataset['en']]        # target sentence
    }

# Now that we've performed the tokenization, it's to create vocabulary.
# vocabulary looks something like dictionary.
"""
vocab: {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    'i': 4,
    'love': 5,
    'my': 6,
    'india': 7,
    ...
}
"""
def build_vocabulary(dataset: list[list[str]], min_freq:int=2):
    vocabulary = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }

    all_tokens_in_dataset = []
    current_idx = 4
    for sentence in dataset:        # ['i', 'love', 'you'], ['i', 'love', 'my', 'india']
        for word in sentence:       # 'i', 'love', 'you'
            all_tokens_in_dataset.append(word)
    token_count = Counter(all_tokens_in_dataset)

    for token, freq in token_count.items():
        if freq >= min_freq:
            vocabulary[token] = current_idx
            current_idx += 1
    return vocabulary

# Next step is to convert each token into number
# auxiliary function to convert token2idx vocabulary to idx2token vocabulary
def swap_k_v_dict(source_dict: dict):
    return { v:k  for k, v in source_dict.items()  }

def convert_into_numbers(tokens: list, token2idx: dict):
    unk_idx = token2idx['<UNK>']
    return [
        token2idx.get(token, unk_idx) for token in tokens
    ]

def numericalization(tokens: list[str], token2idx: dict):
    return (
        [token2idx['<SOS>']]    # SOS token [1]
        + convert_into_numbers(tokens, token2idx=token2idx)     # converts our actual tokens [5,6,7,8,9]
        + [token2idx['<EOS>']]      # EOS token [2]   
    )

# last step of this pre processing pipeline is padding, for this we use builtin library function
# first we'll convert these integer into pytorch tensors and then perform padding using library function
def collate_function(batch, src_pad_token_idx_value=0, tgt_pad_token_idx_value=0):
    src = [
        torch.tensor(item['source']) for item in batch
    ]
    tgt = [
        torch.tensor(item['target']) for item in batch
    ]

    source_padded = pad_sequence(src, batch_first=True, padding_value=src_pad_token_idx_value)
    target_padded = pad_sequence(tgt, batch_first=True, padding_value=tgt_pad_token_idx_value)

    return source_padded, target_padded

# begin with writing the complete execution pipeline, step by step by utilizing above function
# auxiliary function to better readability, DRY principle
def tokenize(dataset):
    return tokenize_function(dataset)

def numericalize(sentences, vocabulary=None):
    sentences_into_numbers = []
    for sentence in sentences:
        sentences_into_numbers.append(
            numericalization(tokens=sentence, token2idx=vocabulary)
        )
    return sentences_into_numbers

# step 1: separate training, validation and testing datasets
train_dataset, validation_dataset, test_dataset = raw_dataset['train'], raw_dataset['validation'], raw_dataset['test']
print(train_dataset.shape, validation_dataset.shape, test_dataset.shape)

# step 2: tokenize source and target data
print("Begin tokenization of all 3 set's")
tokenize_sentences = tokenize(train_dataset)   # this will return tokenized sentences for both source and target.
tokenize_sentences_validation = tokenize(validation_dataset)   # exact same for validation and testing datasets
tokenize_sentences_testing = tokenize(test_dataset)

# step 3: create vocabulary
print("Building Vocabulary")
source_vocabulary = build_vocabulary(tokenize_sentences['source'])
target_vocabulary = build_vocabulary(tokenize_sentences['target'])

print("length of my source vocabulary :", len(source_vocabulary))
print("length of my target vocabulary :", len(target_vocabulary))

# step 4: numericalization
print("Numericalizing ...")

class TranslationDataset(Dataset):
    def __init__(self,
                source_sentences: list[list[int]],
                target_sentences: list[list[int]]):
        assert (
            len(source_sentences) == len(target_sentences)
        ), "source and target must have same no. of sentences"
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self,):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return {
            'source': self.source_sentences[idx],
            'target': self.target_sentences[idx]
        }

numericalize_sentences = {
    'source': numericalize(tokenize_sentences['source'], vocabulary=source_vocabulary),
    'target': numericalize(tokenize_sentences['target'], vocabulary=target_vocabulary),
}

numericalize_sentences_validation = {
    'source': numericalize(tokenize_sentences_validation['source'], vocabulary=source_vocabulary),
    'target': numericalize(tokenize_sentences_validation['target'], vocabulary=target_vocabulary)
}

numericalize_sentences_testing = {
    'source': numericalize(tokenize_sentences_testing['source'], vocabulary=source_vocabulary),
    'target': numericalize(tokenize_sentences_testing['target'], vocabulary=target_vocabulary)
}

train_data = TranslationDataset(numericalize_sentences['source'], numericalize_sentences['target'])
validation_data = TranslationDataset(numericalize_sentences_validation['source'], numericalize_sentences_validation['target'])
test_data = TranslationDataset(numericalize_sentences_testing['source'], numericalize_sentences_testing['target'])

# step 5: batching and padding
batch_size = 64
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_function
)

# exact same for validation and testing datasets
validation_loader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_function
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_function
)

# encoder hyper-parameters
INPUT_DIM = len(source_vocabulary)
OUTPUT_DIM = len(target_vocabulary)

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256

HID_DIM = 512
N_LAYERS = 2

ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

# model creation
enc = MyCustomEncoder(INPUT_DIM,
                    ENC_EMB_DIM,
                    HID_DIM,
                    N_LAYERS,
                    ENC_DROPOUT,
                    DEVICE)
                    
dec = MyCustomDecoder(OUTPUT_DIM,
                    DEC_EMB_DIM,
                    HID_DIM,
                    N_LAYERS,
                    DEC_DROPOUT,
                    )

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# weight initialization of model parameter
# because a random weight initialization can de-stablize the training, that's why we're doing uniform weight initialization
def init_weight(m):
    for name, parameter in m.named_parameters():
        nn.init.uniform_(parameter.data, -0.08, 0.08)

# defining optimizer
optimizer = optim.Adam(model.parameters())

# defining loss function
PAD_IDX = target_vocabulary['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# model training loop
def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for source, target in dataloader:
        source, target = source.to(device=DEVICE), target.to(device=DEVICE)

        optimizer.zero_grad()
        output = model(source, target)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss/len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for source, target in dataloader:
            source, target = source.to(device=DEVICE), target.to(device=DEVICE)
            output = model(source, target)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()
    model.train()
    return epoch_loss/len(dataloader)

N = 15
CLIP = 1

print("Starting Training ...")

for epoch in range(N):
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    validation_loss = evaluate(model, validation_loader, criterion)
    end_time = time.time()

    epoch_mins = int((end_time-start_time)/60)
    epoch_secs = int((end_time-start_time-epoch_mins*60))

    print(f"Epoch: {epoch+1:.2f} | Time: {epoch_mins}m:{epoch_secs}s")
    print(f"    Train Loss: {train_loss:.3f}    |   Train PPL: {math.exp(train_loss):7.3f}")
    print(f"    Val. Loss: {validation_loss:.3f}    |   Validation PPL: {math.exp(validation_loss):7.3f}")
