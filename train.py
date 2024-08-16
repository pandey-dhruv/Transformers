import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
#this library allows you to create absolute paths given relative paths
from pathlib import Path
from dataset import causal_mask, BilingualDataset
from transformer import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import get_weights_file_path, get_config
import warnings



def greedy_decode(model, source, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    #first we define the sos and eos tokens that are required here
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    #precompute the encoder input and it is fed (the same) to the decoder at every decode step
    encoder_output = model.encode(source, src_mask).to(device)
    assert encoder_output.get_device() == 0
    #here we make an empty tensor of shape (1,1) and fill sos_idx to the value. The type is same as that of the source and copied to device
    #the first dimension is for batch_Size and the second dimension is for len of the sequence
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        #we only use the last index of the values calculated becuase they contain the context of the entire sentence
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word == eos_idx:
            break
        
    assert decoder_input.squeeze(0).get_device() == 0
    return decoder_input.squeeze(0)  #only return an array containing the answer to all the input english statements
    #essentially removing the batch dimension from the tensor
        

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    #this to tell the model to be in te evaluation state. Thus the dropout layers do not actually switch off some neurons during evaluation
    #also prepares the batch normalization and the other layers for evaluation because we do not want to update their values during evaluation
    source_texts = []
    expected = []
    predicted = []
    console_width = 80
    count = 0
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  #need to copy the tensors to the GPUS memory
            encoder_mask = batch['encoder_mask'].to(device)
            #we calculate the encoder output only once and use it for the entire decode sequence (all the tokens that are generated)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"  #this is the message displayer if assert fails
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            assert model_out.get_device() == 0
            
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")
            if count == num_examples:
                break
            
        
    model.eval()
def get_all_sentences(ds, lang):
    #the dataset contains a translation column and each row in the column is a dictionary of sorts
    #there "en": contains the english sentence
    #"fr" contains the corresponding french sentence
    for item in ds:
        yield item['translation'][lang]
        
        
#this creates a tokenizer for the model config, the dataset being used and the language (because the tokenization would depend upon the language too)
def get_or_build_tokenizer(config, ds, lang):
    #think of this as config['tokenizer_file'] is a string that can be formatted with a variable
    #like config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json' which can be formatted using .format()
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        #if the above path does not exist, we create it
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))  #if a word is not in the vocab, we use the unknown token
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2)  
        #for a word to be added to the vocabulary, it should hava a minimum frequency of 2
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    #this allows us to split the given dataset (ds_raw here) into smaller datasets of the given size. Here it will be divided into two
    #smaller datasets of size 0.9*len(ds) and the remaining as the validation dataset
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    #to get the maximum length of the sentences in the dataset. This is to ensure that the sequence length that we feed into the
    #model is not too small so that it can conver all the sentences in the dataset
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of the source sentences: {max_len_src}")
    print(f"Max length of the target sentences: {max_len_tgt}")
    #ig we write the code for a single GPU processing core and thus we keep the batch dimension value as 1 in our case.
    #so we write the code as if it is for a single training example because actually, only a single training example is fed to the model
    #at a time step. The examples are fed in parallel to different GPU processing cores that run the model in parallel.
    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=False)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    #set the device where we will load all the tensors.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state["model_state_dict"])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            #we copy the tensors to the device that is used to process the model
            encoder_input = batch['encoder_input'].to(device)   #(batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device)   #(batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     #(batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     #(batch_size, 1, seq_len, seq_len)
            
            assert encoder_input is not None
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device)
            
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            #log the loss calculated above
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            #backpropagate the loss
            loss.backward()
            
            #update the wweights
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "epoch":epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
            