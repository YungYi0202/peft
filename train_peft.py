# Run without int8, with fp16
from huggingface_hub import notebook_login

notebook_login()

import inspect
import random
import sys

import nlp2
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer
from transformers import Trainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from datasets import load_from_disk

from module.args import parse_args
from module.data_processing import (
    encode_dataset,
    DataCollatorCTCWithPadding,
    prepare_dataset_hf,
    prepare_dataset_custom,
    prepare_dataset_whisper,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from module.metric import cer_cal, wer_cal, postprocess
from module.utility import FreezingCallback, SavePeftModelCallback, LrRescheduleTrainer

from datetime import datetime

from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig
from peft import prepare_model_for_int8_training

import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import torch

def load_peft_model_from_hub(peft_model_id):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, peft_model_id) 
    
    print("Load model from hub successfully.")
    return model

def train_n_evaluate(epoch_cnt, model, input_arg, processor, data_collator, data_train, data_test, time):
    ################
    #     Train    #
    ################
    print("***************")
    print(f'Training epoch: {epoch_cnt}')
    print("***************")


    if input_arg.get("sweep_split_shard", False):
        shuffled_dataset = data_train.shuffle(seed=42)
        data_train = shuffled_dataset.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_train = data_train.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_test = data_train


    training_args = Seq2SeqTrainingArguments(
        output_dir=input_arg.get("output_dir", input_arg["repo_name"]),
        length_column_name="lengths",
        group_by_length=input_arg["group_by_length"],
        per_device_train_batch_size=int(input_arg["batch"]),
        per_device_eval_batch_size=int(input_arg["batch"]),
        gradient_accumulation_steps=int(input_arg["grad_accum"]),
        eval_accumulation_steps=int(input_arg["grad_accum"]),
        evaluation_strategy="epoch",
        save_strategy="no",
        ddp_find_unused_parameters=True,
        resume_from_checkpoint=input_arg.get("checkpoint", False),
        overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
        # load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="cer",
        num_train_epochs=1, # Specified here. The actual epoch is set in trainer.
        fp16=True,
        logging_steps=input_arg.get("logging_steps", 10),
        learning_rate=input_arg.get("learning_rate", 2.34e-4),
        # learning_rate=input_arg.get("learning_rate", 1e-3),
        warmup_steps=input_arg.get("warmup_steps", 100),
        # warmup_steps=input_arg.get("warmup_steps", 50),
        save_total_limit=input_arg.get("save_total_limit", 3),
        push_to_hub=False,
        report_to="all",
        weight_decay=input_arg.get("weight_decay", 0.02),
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    training_args.generation_max_length = 225

    trainer = LrRescheduleTrainer(
        specified_epoch=epoch_cnt,
        total_epoch=input_arg['epoch'],
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Push to Hub    
    peft_model_id = f"{input_arg[hub_username]}/" + f"tw-zh2.6-{input_arg['model_config']}-Lora".replace("/", "-")
    peft_model_id += f"-epoch{epoch_cnt+1)}-total{input_arg['epoch']}epoch"
    print(f"peft_model_id: {peft_model_id}")

    if not input_arg.get("only_eval", False):
        trainer.train(input_arg.get("checkpoint", None))   
        try:
            model.push_to_hub(peft_model_id, use_auth_token=input_arg["hub_use_auth_token"])
        except:
            trainer.save_model(input_arg.get("output_dir", input_arg["repo_name"]))
            print(f'Save model locally.: {input_arg.get("output_dir", input_arg["repo_name"])}')
    elif input_arg.get("checkpoint", None) is None:
        model = load_peft_model_from_hub(peft_model_id)
    
    ###################
    #     Evaluate    #
    ###################
    eval_dataloader = DataLoader(data_test, batch_size=int(input_arg["batch"]), collate_fn=data_collator)
    
    model.eval()
    model = model.to("cuda")
    label_list = []
    pred_list = []
    pred_results = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    # input_features=batch["input_features"],
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    # decoder_input_ids=batch["labels"][:, :4],
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            pred_str = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
            pred_results += pred_result

            pred_list += pred_str
            label_list += label_str

            if step == 0:
                print(pred_result)
        

        del generated_tokens, labels, batch
        gc.collect()

    nlp2.write_csv(pred_results, f'pred_{time}_epoch_{epoch_cnt}.csv')
    cer = cer_cal(label_list, pred_list)
    wer = wer_cal(label_list, pred_list)
    print("********* Evaluation Result *********")
    print(f"cer: {cer}, wer: {wer}")
    print("*************************************")

    model.train()
    return model


def main(arg=None):
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    ############
    #  Config  #
    ############
    
    size = "large-v2"
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    input_arg["train_data"] = "/work/hungyi2022/taiwanese-meta/taiwanese-meta-2.6sec-train.csv"
    input_arg["test_data"] = "/work/hungyi2022/taiwanese-meta/taiwanese-meta-2.6sec-eval.csv"
    input_arg["tokenize_config"] = f"openai/whisper-{size}"
    input_arg["model_config"] = f"openai/whisper-{size}"
    input_arg["output_dir"] = f"outputs/{time}"
    input_arg["group_by_length"] = True
    input_arg["cache_dir"] = '/work/hungyi2022/.cache'
    input_arg["repo_name"] = f"/work/hungyi2022/peft/{input_arg['model_config']}-TW-meta-2.6"
   
    print("input_arg", input_arg)
    
    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], 
        task="transcribe", 
        language="chinese",
        dropout=input_arg.get("dropout", 0.0)
        )
    processor.save_pretrained(input_arg["repo_name"])

    audio_feature_key = "input_ids" if size == "large-v2" else inspect.getfullargspec(model.forward).args[1]
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    
    if input_arg.get('checkpoint', None) is not None:
        model = load_peft_model_from_hub(input_arg.get('checkpoint', None))
    else:
        model = WhisperForConditionalGeneration.from_pretrained(input_arg["model_config"])
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
    

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.print_trainable_parameters()

    ############
    #  Dataset #
    ############

    if not input_arg.get("load_data_from_cache", False):
        # data set
        dataset = load_dataset(
            "csv", 
            data_files=input_arg["train_set"], 
            cache_dir=input_arg["cache_dir"], 
            # cache_dir=None, 
            )
        dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))
        if "custom_set_test" in input_arg:
            dataset_test = load_dataset(
                "csv", 
                data_files=input_arg["custom_set_test"], 
                cache_dir=input_arg["cache_dir"],
                # cache_dir=None,
            )
            dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e["path"]))
            data_test = dataset_test["train"]
        else:
            dataset = dataset["train"].train_test_split(test_size=0.1)
            data_test = dataset["test"]

        data_train = dataset["train"]
        data_train = data_train.map(
            prepare_dataset_whisper,
            num_proc=input_arg["num_proc"],
            fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        )
        data_test = data_test.map(
            prepare_dataset_whisper,
            num_proc=input_arg["num_proc"],
            fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        )

        # original code
        print("before filtering audio length")
        print("data train", data_train)
        print("data test", data_test)
        if input_arg.get("max_input_length_in_sec", None):
            max_input_length_in_sec = input_arg["max_input_length_in_sec"]
            min_input_length_in_sec = 1
            data_train = data_train.filter(
                lambda x: min_input_length_in_sec * processor.feature_extractor.sampling_rate
                < x
                < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
                input_columns=["lengths"],
            )
            data_test = data_test.filter(
                lambda x: min_input_length_in_sec * processor.feature_extractor.sampling_rate
                < x
                < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
                input_columns=["lengths"],
            )
        print("after filtering audio length")
        print("data train", data_train)
        print("data test", data_test)

        # ======================================= #

        print("before filtering label length")
        print("data train", data_train)
        print("data test", data_test)
        data_train = data_train.filter(lambda x: x is not None and 0 < len(x), input_columns=["labels"])
        data_test = data_test.filter(lambda x: x is not None and 0 < len(x), input_columns=["labels"])
        print("after filtering label length")
        print("data train", data_train)
        print("data test", data_test)

        # ======================================= #
        # subprocess.run("rm -rf /home/hungyi2022/.cache", shell=True, check=True)

        print("before encoding dataset")
        print("data train", data_train)
        print("data test", data_test)

        if not input_arg.get("only_eval", False):
            data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})
            data_train.save_to_disk(f"{input_arg["repo_name"]}-train.data")
        
        data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor})
        if not input_arg.get("only_eval", False):
            data_test.save_to_disk(f"{input_arg["repo_name"]}-test.data")
        print("after encoding dataset")
        print("data train", data_train)
        print("data test", data_test)
    else:
        data_train = load_from_disk(f"{input_arg["repo_name"]}-train.data")
        data_test = load_from_disk(f"{input_arg["repo_name"]}-test.data")
    

    print("finalize dataset")
    print("data train", data_train)
    print("data test", data_test)

    ##########################
    #     Train & Evaluate   #
    ##########################
    for i in range(input_arg['epoch']):
        model = train_n_evaluate(i, model, input_arg, processor, data_collator, data_train, data_test, time)

    
if __name__ == "__main__":
    main()
