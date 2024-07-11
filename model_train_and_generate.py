''' ------------------------------------------------------------------------
                                imports                              
    ------------------------------------------------------------------------
'''
# basic python and machine learning libraries
import numpy as np
import pandas as pd
import os
import sys
import json

# helper file import
from helper import  Helper 
from huggingface_hub import login

# Connect google drive to colab
from google.colab import drive

# Lirbraries specific to the fine-tuning and optimization of the large language models
import torch
import evaluate
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

def model_train_generate_save(database='adventureworks', window=1, test_file= 'products', model='meta-llama/Llama-2-7b-hf'):
    ''' ------------------------------------------------------------------------
                    intial model, device setup, and authentication                              
        ------------------------------------------------------------------------
    '''
    model_name = "mistralai/Mistral-7B-v0.1"                             # base model
    device = 'cuda'                                                      # set device - we used GPU A100 and supported CUDA
    login()                                                              # insert your access token - especially if using gated models 

    ''' ------------------------------------------------------------------------
                                   tokenizer configuration                              
        ------------------------------------------------------------------------
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # Tokenizer
    # Redefined the pad_token and pad_token_id with end-of-string token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
  
    ''' ------------------------------------------------------------------------
                            Model Optimization                              
        ------------------------------------------------------------------------
    '''
    ''' ------------------------------------------------------------------------
                            quantization configuration                              
        ------------------------------------------------------------------------
    '''

    # Details on quantization in - https://huggingface.co/docs/optimum/concept_guides/quantization
    compute_dtype = getattr(torch, "float16")
    print(compute_dtype)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )

    ''' ------------------------------------------------------------------------
                        configuration - quantize and load model                               
        ------------------------------------------------------------------------
    '''
    model = AutoModelForCausalLM.from_pretrained(
              model_name,
              quantization_config=bnb_config,
              attn_implementation="flash_attention_2", # only if using GPU A100
              device_map={"": 0},                      #device_map="auto" caused a problem in the training

    )

    #print(model)                                      # can see all the layers in Linear4bit

    ''' ------------------------------------------------------------------------
                            PEFT - LoRA configuration                              
        ------------------------------------------------------------------------
    '''

    # Details on PEFT - LoRA in - https://huggingface.co/papers/2309.15223 
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,           # rank increases 
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head",]
    )

    model = prepare_model_for_kbit_training(model)          # Casts some modules of the model to fp32
    model.config.pad_token_id = tokenizer.pad_token_id      # Configure the pad token in the model
    model.config.use_cache = False                          # Gradient checkpointing is used by default but not compatible with caching



    ''' ------------------------------------------------------------------------
                             Model Training Arguments                              
        ------------------------------------------------------------------------
    '''

    training_arguments = TrainingArguments(
            output_dir="./results",         # directory in which the checkpoint will be saved.
            eval_strategy="steps",          # default = 'no'. other options - epoch
            optim="paged_adamw_8bit",       # optimizer best with QLoRA. detailes - https://huggingface.co/docs/transformers/perf_train_gpu_one#8-bit-adam
            per_device_train_batch_size=8,  # batch size for training
            per_device_eval_batch_size=8,   # batch size for evaluation
            gradient_accumulation_steps=2,  # number of lines to accumulate gradient. it changes the size of a "step". details - https://huggingface.co/docs/transformers/main/en/main_classes/&amp;num;transformers.TrainingArguments.gradient_accumulation_steps
            gradient_checkpointing=True,    # details in - https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
            log_level="debug",              # other options - ‘info’, ‘warning’, ‘error’ and ‘critical’ . default 'passive' -> 'warning'
            save_steps=150,                 # number of steps between checkpoints
            logging_steps=50,               # number of steps between logging of the loss for monitoring
            learning_rate=4e-4,             # initial learning rate for AdamW optimizer
            fp16=True,                      # to use fp16 16-bit(mixed) precision training
            num_train_epochs=5,             # number of training epochs
            warmup_ratio=0.01,              # Ratio of total training steps used for a linear warmup from 0 to learning_rate. 
            weight_decay=0.01,              # weight decay to apply in optimizer. details - https://huggingface.co/docs/transformers/en/main_classes/&amp;num;transformers.TrainingArguments.weight_decay
            lr_scheduler_type="linear",     # scheduler type to use. details - https://huggingface.co/docs/transformers/main/en/main_classes/&amp;num;transformers.TrainingArguments.lr_scheduler_type
            report_to="tensorboard"         # report the results and logs to. available options - https://huggingface.co/docs/transformers/main/en/main_classes/&amp;num;transformers.TrainingArguments.report_to
    )

    ''' ------------------------------------------------------------------------
                            Data Connection and Loading                         
        ------------------------------------------------------------------------
    '''
    drive.mount('/content/gdrive')      # mount google drive
    train_data_file_path = f"./data/{database}/window_{window}/train.csv" # custom training dataset
    val_data_file_path = f"./data/{database}/window_{window}/val.csv"     # custom validation dataset

    # Load the dataset
    dataset = load_dataset('csv', data_files={'train': [train_data_file_path], 'validation': [val_data_file_path]}, column_names=['prompt', 'response'])



    # implementation - incomplete
    test_file_path = f"./data/{database}/window_{window}/test/{test_file}.csv"
    test_dataset = load_dataset('csv', data_files={'test': [test_file_path]}, column_names=['prompt', 'response'])
    eval_prompt = Helper.formatting_test_prompts_func(test_dataset['test'])


    ''' ------------------------------------------------------------------------
                            trainer initialization                             
        ------------------------------------------------------------------------
    '''

    trainer = SFTTrainer(
            model=model,                                    # initialized quantized model
            train_dataset=dataset['train'],                 # initialized training dataset
            eval_dataset=dataset['validation'],             # initialized validation dataset
            peft_config=peft_config,                        # initialized parameterized fine-tuning configurations
            formatting_func=Helper.formatting_prompts_func, # formatting function to be used for creating the ConstantLengthDataset
            max_seq_length=512,                             # maximum length of the sequence
            tokenizer=tokenizer,                            # initialized tokenizer
            args=training_arguments,                        # initialized training arguments for fine-tuning
    )

    ''' ------------------------------------------------------------------------
                            Training - fine-tuning                             
        ------------------------------------------------------------------------
    '''
    trainer.train()                                         # training

    ''' ------------------------------------------------------------------------
                            Generate Output                             
        ------------------------------------------------------------------------
    '''
    model_outputs = []
    for test_data in eval_prompt:
        model_input = tokenizer(test_data, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = tokenizer.decode(model.generate(**model_input, max_length=4096,tokenizer=tokenizer, stop_strings='</root>', pad_token_id=2)[0], skip_special_tokens=True)
            model_outputs.append(output)

    pd.DataFrame(model_outputs).to_csv("/content/gdrive/MyDrive/path/to/csv/data/file")


    ''' ------------------------------------------------------------------------
                               Save Model                             
        ------------------------------------------------------------------------
    '''

    ''' Saving 4-bit quantized model is still not supported yet. 
        That is why we have to merge the trained adapter to the base model.
        To learn more, please visit https://github.com/huggingface/transformers/issues/23904
    '''

    new_model = 'Mistral7B_NorthWind'
    trainer.model.save_pretrained(new_model)

    #Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # compares the weights and biases of the base model to the trained model and merges the fine-tuned parameters
    peft_model = PeftModel.from_pretrained(base_model, new_model)
    merged_model = peft_model.merge_and_unload()

    output_merged_dir = f"./models/{new_model}_winodw_{window}"

    os.makedirs(output_merged_dir, exist_ok=True)
    merged_model.save_pretrained(output_merged_dir, safe_serialization = False)
    # save_pretrained is the base class for the all the variants like save_model(), save() etc. in the huggingFace stack now
    tokenizer.save_pretrained(output_merged_dir)





def main():
    database = 'northwind'
    window = 4
    test_file = 'products'
    model = 'meta-llama/Llama-2-7b-hf'

    model_train_generate_save(database, window, test_file, model)

if __name__=="__main__":
    main()
