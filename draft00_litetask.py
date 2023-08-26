# -*- coding: utf-8 -*-

# !pip install --upgrade youtube_dl
# !sudo pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
# !apt -y install ffmpeg lame
# !pip install bitsandbytes
# !pip install accelerate
# !pip install -q transformers datasets librosa evaluate jiwer gradio bitsandbytes==0.37 accelerate
# !pip install -q git+https://github.com/huggingface/peft.git@main
# !pip install -U openai-whisper

class draft00_litetask():
    def __init__(self):
        pass

    def run(self, msg):
		# ЧЕРНОВОЙ ВАРИАНТ ПО ЗАПУСКУ WHISPER LARGE V2 ИЗ COLAB

		# from google.colab import drive
		# drive.mount('/content/drive')
		

		"""## ЗАРЕГИСТРИРОВАЛИСЬ НА HUGGING FACE, ИЗ SETTINGS, ACCESS TOKENS СОЗДАЛИ ТОКЕН"""

		from huggingface_hub import notebook_login

		notebook_login()



		"""## CОХРАНИЛИ В MP4 ФАЙЛЫ"""

		# README https://github.com/ryanwebster90/colab-yt-dl/blob/main/dl_yt_playlist.ipynb



		# README https://stackoverflow.com/a/76409717



		yt_urls = ['https://www.youtube.com/watch?v=cEeHV7zzOv8&list=PLMquADD-aVhqY63PQl0DBETkI-cguhzk5&index=1&pp=iAQB',
				   'https://www.youtube.com/watch?v=aI71FGde_0k&list=PLMquADD-aVhqY63PQl0DBETkI-cguhzk5&index=2&pp=iAQB',
				   'https://www.youtube.com/watch?v=Kx6pJDzqMtI&list=PLMquADD-aVhqY63PQl0DBETkI-cguhzk5&index=3&pp=iAQB']
		import os
		def my_mkdirs(folder):
		  if os.path.exists(folder)==False:
			os.makedirs(folder)
		my_mkdirs('/content/tmp/')
		output_folder = '/content/drive/My Drive/kia/ambient20/'
		my_mkdirs(output_folder)

		# download youtube videos
		for ind,url in enumerate(yt_urls):
		  !youtube-dl $url -f 'bestaudio[ext=m4a]' -o 'tmp/%(title)s.m4a'
		# !youtube-dl https://www.youtube.com/watch?v=Mhi6Lb52ZbM -f 'bestaudio[ext=m4a]' -o tmp3.m4a

		youtube_files = []

		import glob
		files = glob.glob('/content/tmp/*')
		for file in files:
		  out_file = f'{output_folder}{file[13:-3]}.mp3'
		  file = file.replace(' ','\ ')
		  out_file = out_file.replace(' ','\ ')
		  youtube_files.append(out_file)
		  !ffmpeg -i $file -vn -ab 128k -ar 44100 -y $out_file

		# README https://github.com/Vaibhavs10/fast-whisper-finetuning/blob/main/Whisper_w_PEFT.ipynb

		# README https://colab.research.google.com/drive/1DOkD_5OUjFa0r5Ik3SgywJLJtEo2qLxO?usp=sharing#scrollTo=090fa3ed

		"""## ДАЛЕЕ ЭКСПЕРИМЕНТЫ"""








		"""# Fine-tune Whisper (large) with LoRA & BNB powerd by PEFT ⚡️

		A one size fits all notebook, to fine-tune Whisper (large) on a consumer GPU with less than 8GB GPU VRAM, all with comparable performance to full-finetuning. ⚡️

		We present a step-by-step guide on how to fine-tune Whisper with Common Voice 13.0 dataset using 🤗 Transformers and PEFT. In this Colab, we leverage `PEFT` and `bitsandbytes` to train a `whisper-large-v2` checkpoint seamlessly with a free T4 GPU (16 GB VRAM).

		For more details on Whisper fine-tuning, datasets and metrics, refer to Sanchit Gandhi's brilliant blogpost: [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)

		## Why Parameter Efficient Fine Tuning ([PEFT](https://github.com/huggingface/peft))?

		As the model size continue to increase, fine tuning a model has become both computationally expensive and storage heavy. For example, a `Whisper-large-v2` model requires ~24GB of GPU VRAM to fine-tune for full fine-tuning and requires ~7 GB of storage for each fine-tuned storage. For low-resource environments this becomes quite a bottleneck and often near impossible to get meaningful results.

		Cue, PEFT, with PEFT you can tackle this bottleneck head on. PEFT approaches (like Low Rank Adaptation) only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained model, thereby greatly decreasing the computational and storage costs. We've observed that it also overcomes the issues of catastrophic forgetting, a behaviour observed during the full finetuning of large models.

		### Aha! So wait, what's this LoRA thing?

		PEFT comes out-of-the-box with multiple parameter efficient techniques. One such technique is [Low Rank Adaptation or LoRA](https://github.com/microsoft/LoRA). LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This greatly reduces the number of trainable parameters for downstream tasks.

		LoRA performs on-par or better than fine-tuning in model quality despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.

		### That's all cool, but show me the numbers?

		Don't worry, we got ya! We ran multiple experiments to compare a full fine-tuning of Whisper-large-v2 checkpoint and that with PEFT, here's what we found:

		1. We were able to fine-tune a 1.6B parameter model with less than 8GB GPU VRAM. 🤯
		2. With significantly less number of traininable parameters, we were able to fit almost **5x** more batch size. 📈
		3. The resultant checkpoint were less than 1% the size of the original model, ~60MB (i.e. 1% the size of orignal model) 🚀

		To make things even better, all of this comes with minimal changes to the existing 🤗 transformers Whisper inference codebase.

		### Curious to test this out for yourself? Follow along!

		## Prepare Environment

		We'll employ several popular Python packages to fine-tune the Whisper model.
		We'll use `datasets` to download and prepare our training data and
		`transformers` to load and train our Whisper model. We'll also require
		the `librosa` package to pre-process audio files, `evaluate` and `jiwer` to
		assess the performance of our model. Finally, we'll
		use `PEFT`, `bitsandbytes`, `accelerate` to prepare and fine-tune the model with LoRA.
		"""



		"""With the environment now set up, let's try to secure a decent GPU for our Colab! Unfortunately, it's becoming much harder to get access to a good GPU with the free version of Google Colab. However, with Google Colab Pro one should have no issues in being allocated a V100 or P100 GPU.

		To get a GPU, click _Runtime_ -> _Change runtime type_, then change _Hardware accelerator_ from _None_ to _GPU_.

		We can verify that we've been assigned a GPU and view its specifications:
		"""

		gpu_info = !nvidia-smi
		gpu_info = '\n'.join(gpu_info)
		if gpu_info.find('failed') >= 0:
		  print('Not connected to a GPU')
		else:
		  print(gpu_info)

		"""Alrighty! Let's configure our environment to ensure it uses the GPU provided by Colab to us."""

		import os

		os.environ["CUDA_VISIBLE_DEVICES"] = "0"

		"""We strongly advise you to upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/)
		whilst training. The Hub provides:
		- Integrated version control: you can be sure that no model checkpoint is lost during training.
		- Tensorboard logs: track important metrics over the course of training.
		- Model cards: document what a model does and its intended use cases.
		- Community: an easy way to share and collaborate with the community!

		Linking the notebook to the Hub is straightforward - it simply requires entering your Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):
		"""

		# FIXME from huggingface_hub import notebook_login

		# FIXME notebook_login()

		"""Next up, we define Whisper model checkpoints and task details."""

		# FIXME model_name_or_path = "openai/whisper-large-v2"
		# FIXME task = "transcribe"

		"""Lastly, we define the dataset details, including the language we'd like to fine-tune Whisper on too."""

		# FIXME dataset_name = "mozilla-foundation/common_voice_13_0"
		# FIXME language = "Hindi"
		# FIXME language_abbr = "hi" # Short hand code for the language we want to fine-tune

		"""# Load Dataset

		Using 🤗 Datasets, downloading and preparing data is extremely simple.
		We can download and prepare the Common Voice splits in just one line of code.

		First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_13_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.

		Since Hindi is very low-resource, we'll combine the `train` and `validation`
		splits to give approximately 12 hours of training data. We'll use the 6 hours
		of `test` data as our held-out test set:
		"""

		# FIXME from datasets import load_dataset, DatasetDict

		# FIXME common_voice = DatasetDict()

		# FIXME common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
		# FIXME common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

		# FIXME print(common_voice)

		"""Most ASR datasets only provide input audio samples (`audio`) and the
		corresponding transcribed text (`sentence`). Common Voice contains additional
		metadata information, such as `accent` and `locale`, which we can disregard for ASR.
		Keeping the notebook as general as possible, we only consider the input audio and
		transcribed text for fine-tuning, discarding the additional metadata information:
		"""

		# FIXME common_voice = common_voice.remove_columns(
		# FIXME     ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
		# FIXME )

		# FIXME print(common_voice)

		"""## Prepare Feature Extractor, Tokenizer and Data

		The ASR pipeline can be de-composed into three stages:
		1. A feature extractor which pre-processes the raw audio-inputs
		2. The model which performs the sequence-to-sequence mapping
		3. A tokenizer which post-processes the model outputs to text format

		In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer,
		called [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor)
		and [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer)
		respectively.
		"""

		# FIXME from transformers import WhisperFeatureExtractor

		# FIXME feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

		# FIXME from transformers import WhisperTokenizer

		# FIXME tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

		"""To simplify using the feature extractor and tokenizer, we can _wrap_ both into a single `WhisperProcessor` class. This processor object can be used on the audio inputs and model predictions as required.
		In doing so, we only need to keep track of two objects during training:
		the `processor` and the `model`:
		"""

		# FIXME from transformers import WhisperProcessor

		# FIXME processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

		"""### Prepare Data

		Let's print the first example of the Common Voice dataset to see
		what form the data is in:
		"""

		# FIXME print(common_voice["train"][0])

		"""Since
		our input audio is sampled at 48kHz, we need to _downsample_ it to
		16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

		We'll set the audio inputs to the correct sampling rate using dataset's
		[`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
		method. This operation does not change the audio in-place,
		but rather signals to `datasets` to resample audio samples _on the fly_ the
		first time that they are loaded:
		"""

		# FIXME from datasets import Audio

		# FIXME common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

		"""Re-loading the first audio sample in the Common Voice dataset will resample
		it to the desired sampling rate:
		"""

		# FIXME print(common_voice["train"][0])

		"""Now we can write a function to prepare our data ready for the model:
		1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
		2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
		3. We encode the transcriptions to label ids through the use of the tokenizer.
		"""

		# FIXME def prepare_dataset(batch):
		# FIXME     # load and resample audio data from 48 to 16kHz
		# FIXME     audio = batch["audio"]

		# FIXME     # compute log-Mel input features from input audio array
		# FIXME     batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

		# FIXME     # encode target text to label ids
		# FIXME     batch["labels"] = tokenizer(batch["sentence"]).input_ids
		# FIXME     return batch

		"""We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

		Make yourself some tea 🍵, depending on dataset size, this might take 20-30 minutes ⏰
		"""

		# FIXME common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

		# FIXME common_voice["train"]

		"""## Training and Evaluation

		Now that we've prepared our data, we're ready to dive into the training pipeline.
		The [🤗 Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer)
		will do much of the heavy lifting for us. All we have to do is:

		- Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

		- Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric.

		- Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.

		- Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.

		Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it
		to transcribe speech in Hindi.

		### Define a Data Collator

		The data collator for a sequence-to-sequence speech model is unique in the sense that it
		treats the `input_features` and `labels` independently: the  `input_features` must be
		handled by the feature extractor and the `labels` by the tokenizer.

		The `input_features` are already padded to 30s and converted to a log-Mel spectrogram
		of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
		to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.

		The `labels` on the other hand are un-padded. We first pad the sequences
		to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens
		are then replaced by `-100` so that these tokens are **not** taken into account when
		computing the loss. We then cut the BOS token from the start of the label sequence as we
		append it later during training.

		We can leverage the `WhisperProcessor` we defined earlier to perform both the
		feature extractor and the tokenizer operations:
		"""

		# FIXME import torch

		# FIXME from dataclasses import dataclass
		# FIXME from typing import Any, Dict, List, Union


		# FIXME @dataclass
		# FIXME class DataCollatorSpeechSeq2SeqWithPadding:
		# FIXME     processor: Any

		# FIXME     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# FIXME         # split inputs and labels since they have to be of different lengths and need different padding methods
		# FIXME         # first treat the audio inputs by simply returning torch tensors
		# FIXME         input_features = [{"input_features": feature["input_features"]} for feature in features]
		# FIXME         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

				# get the tokenized label sequences
		# FIXME         label_features = [{"input_ids": feature["labels"]} for feature in features]
				# pad the labels to max length
		# FIXME         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

				# replace padding with -100 to ignore loss correctly
		# FIXME         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

				# if bos token is appended in previous tokenization step,
				# cut bos token here as it's append later anyways
		# FIXME         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
		# FIXME             labels = labels[:, 1:]

		# FIXME         batch["labels"] = labels

		# FIXME         return batch

		"""Let's initialise the data collator we've just defined:"""

		# FIXME data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

		"""### Evaluation Metrics

		We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing
		ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:
		"""

		# FIXME import evaluate

		# FIXME metric = evaluate.load("wer")

		"""### Load a Pre-Trained Checkpoint

		Now let's load the pre-trained Whisper checkpoint. Again, this
		is trivial through use of 🤗 Transformers!

		To reduce our models memory footprint, we load the model in 8bit, this means we quantize the model to use 1/4th precision (when comapared to float32) with minimal loss to performance. To read more about how this works, head over [here](https://huggingface.co/blog/hf-bitsandbytes-integration).
		"""

		# FIXME from transformers import WhisperForConditionalGeneration

		# FIXME model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")

		"""### Post-processing on the model

		Finally, we need to apply some post-processing steps on the 8-bit model to enable training. We do so by first freezing all the model layers, and then cast the layer-norm and the output layer in `float32` for training and model stability.
		"""

		# FIXME from peft import prepare_model_for_int8_training

		# FIXME model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")

		"""Since the Whisper model uses Convolutional layers in the Encoder, checkpointing disables grad computation to avoid this we specifically need to make the inputs trainable."""

		# FIXME def make_inputs_require_grad(module, input, output):
		# FIXME     output.requires_grad_(True)

		# FIXME model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

		"""### Apply Low-rank adapters (LoRA) to the model

		Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
		"""

		# FIXME from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

		# FIXME config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

		# FIXME model = get_peft_model(model, config)
		# FIXME model.print_trainable_parameters()

		"""We are ONLY using **1%** of the total trainable parameters, thereby performing **Parameter-Efficient Fine-Tuning**

		### Define the Training Configuration

		In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
		"""

		# FIXME from transformers import Seq2SeqTrainingArguments

		# FIXME training_args = Seq2SeqTrainingArguments(
		# FIXME     output_dir="reach-vb/test",  # change to a repo name of your choice
		# FIXME     per_device_train_batch_size=8,
		# FIXME     gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
		# FIXME     learning_rate=1e-3,
		# FIXME     warmup_steps=50,
		# FIXME     num_train_epochs=1,
		# FIXME     evaluation_strategy="steps",
		# FIXME     fp16=True,
		# FIXME     per_device_eval_batch_size=8,
		# FIXME     generation_max_length=128,
		# FIXME     logging_steps=100,
		# FIXME     max_steps=100, # only for testing purposes, remove this from your final run :)
		# FIXME     remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
		# FIXME     label_names=["labels"],  # same reason as above
		# FIXME )

		"""Fine-tuning a model with PEFT comes with a few caveats.

		1. We need to explicitly set `remove_unused_columns=False` and `label_names=["labels"]` as the PeftModel's forward doesn't inherit the signature of the base model's forward.

		2. Since INT8 training requires autocasting, we cannot use the native `predict_with_generate` call in Trainer as it doesn't automatically cast.

		3. Similarly, since we cannot autocast, we cannot pass the `compute_metrics` to `Seq2SeqTrainer` so we'll comment it out whilst instantiating the Trainer.
		"""

		# FIXME from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
		# FIXME from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

		# This callback helps to save only the adapter weights and remove the base model weights.
		# FIXME class SavePeftModelCallback(TrainerCallback):
		# FIXME     def on_save(
		# FIXME         self,
		# FIXME         args: TrainingArguments,
		# FIXME         state: TrainerState,
		# FIXME         control: TrainerControl,
		# FIXME         **kwargs,
		# FIXME     ):
		# FIXME         checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

		# FIXME         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
		# FIXME         kwargs["model"].save_pretrained(peft_model_path)

		# FIXME         pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
		# FIXME         if os.path.exists(pytorch_model_path):
		# FIXME             os.remove(pytorch_model_path)
		# FIXME         return control


		# FIXME trainer = Seq2SeqTrainer(
		# FIXME     args=training_args,
		# FIXME     model=model,
		# FIXME     train_dataset=common_voice["train"],
		# FIXME     eval_dataset=common_voice["test"],
		# FIXME     data_collator=data_collator,
			# compute_metrics=compute_metrics,
		# FIXME     tokenizer=processor.feature_extractor,
		# FIXME     callbacks=[SavePeftModelCallback],
		# FIXME )
		# FIXME model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

		# FIXME trainer.train()

		"""Now that our model is fine-tuned, we can push the model on to Hugging Face Hub, this will later help us directly infer the model from the model repo."""

		# FIXME peft_model_id = "reach-vb/whisper-large-v2-hindi-100steps"
		# FIXME model.push_to_hub(peft_model_id)

		"""## ВСЕ ДО ЭТОГО ШАГА МОЖНО ПРОПУСТИТЬ И СКАЧАТЬ СРАЗУ ОБУЧЕННУЮ МОДЕЛЬ"""

		# https://github.com/guillaumekln/faster-whisper/issues/36#issuecomment-1480999802

		# I pushed converted models to the HuggingFace Hub: https://huggingface.co/guillaumekln/faster-whisper-large-v2/tree/main

		# The linked PR is also updating the code to automatically download these models so that the conversion step is no longer needed in most cases.

		# README https://huggingface.co/guillaumekln/faster-whisper-large-v2/blob/main/README.md

		# EXAMPLE https://huggingface.co/openai/whisper-large-v2

		# FIXME !git lfs install
		# FIXME !rm -rf openai
		# FIXME !mkdir -p openai
		# FIXME !git clone https://huggingface.co/openai/whisper-large-v2 openai/whisper-large-v2

		# FIXME !git lfs install
		# FIXME !rm -rf guillaumekln
		# FIXME !mkdir -p guillaumekln
		# FIXME !git clone https://huggingface.co/guillaumekln/faster-whisper-large-v2 guillaumekln/faster-whisper-large-v2



		import torch
		from transformers import pipeline
		from datasets import load_dataset

		device = "cuda:0" if torch.cuda.is_available() else "cpu"

		pipe = pipeline(
		  "automatic-speech-recognition",
		  model="openai/whisper-large-v2",
		  chunk_length_s=30,
		  device=device,
		)

		# FIXME ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
		# FIXME sample = ds[0]["audio"]

		# FIXME prediction = pipe(sample.copy(), batch_size=8)["text"]
		# FIXME " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

		# we can also return timestamps for the predictions
		# FIXME prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
		# FIXME [{'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.',
		# FIXME   'timestamp': (0.0, 5.44)}]

		!mkdir -p /content/drive/LARGE_V2

		"""## ОШИБКА - СЕАНС ПРЕКРАЩЕН ИЗ-ЗА НЕХВАТКИ ОЗУ"""

		import whisper

		model = whisper.load_model("large-v2")
		count = 0
		for youtube_file in youtube_files:
			count = count + 1
			with open(f"/content/drive/LARGE_V2/{count}.txt", "w") as fw:
				# load audio and pad/trim it to fit 30 seconds
				audio = whisper.load_audio(youtube_file)
				audio = whisper.pad_or_trim(audio)

				# make log-Mel spectrogram and move to the same device as the model
				mel = whisper.log_mel_spectrogram(audio).to(model.device)

				# detect the spoken language
				_, probs = model.detect_language(mel)
				print(f"Detected language: {max(probs, key=probs.get)}")

				# decode the audio
				options = whisper.DecodingOptions()
				result = whisper.decode(model, mel, options)

				# print the recognized text
				print(result.text)
				fw.write(result.text)

		# FIXME segments, info = model.transcribe(youtube_file)
		# FIXME for segment in segments:
		# FIXME     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

		# FIXME import torch

		# FIXME from transformers import pipeline

		# FIXME from datasets import load_dataset

		# FIXME device = "cuda:0" if torch.cuda.is_available() else "cpu"

		# FIXME pipe = pipeline(

		# FIXME   "automatic-speech-recognition",

		# FIXME   model="openai/whisper-large-v2",

		# FIXME   chunk_length_s=30,

		# FIXME   device=device,

		# FIXME )

		# FIXME ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

		# FIXME sample = ds[0]["audio"]

		# FIXME prediction = pipe(sample.copy(), batch_size=8)["text"]
		# FIXME " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

		# we can also return timestamps for the predictions

		# FIXME prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
		# FIXME [{'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.',
		# FIXME   'timestamp': (0.0, 5.44)}]

		# FIXME from transformers import WhisperProcessor, WhisperForConditionalGeneration

		# FIXME from datasets import Audio, load_dataset

		# load model and processor

		# FIXME processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

		# FIXME model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

		# FIXME forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")

		# load streaming dataset and read first audio sample

		# FIXME ds = load_dataset("common_voice", "ru", split="test", streaming=True)

		# FIXME ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

		# FIXME input_speech = next(iter(ds))["audio"]

		# FIXME input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

		# generate token ids

		# FIXME predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

		# decode token ids to text

		# FIXME transcription = processor.batch_decode(predicted_ids)
		# FIXME ['<|startoftranscript|><|ru|><|transcribe|><|notimestamps|> Наконец-то будет проведена действительно интересная работа по этой теме.<|endoftext|>']

		# FIXME transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
		# FIXME [' Наконец-то будет проведена действительно интересная работа по этой теме.']

		"""## ВСЕ ПОСЛЕ ЭТОГО ШАГА МОЖНО ПРОПУСТИТЬ ТАК КАК СКАЧАЛИ ОБУЧЕННУЮ МОДЕЛЬ

		# Evaluation and Inference

		On to the fun part, we've successfully fine-tuned our model. Now let's put it to test and calculate the WER on the `test` set.

		As with training, we do have a few caveats to pay attention to:
		1. Since we cannot use `predict_with_generate` function, we will hand roll our own eval loop with `torch.cuda.amp.autocast()` you can check it out below.
		2. Since the base model is frozen, PEFT model sometimes fails to recognise the language while decoding. To fix that, we force the starting tokens to mention the language we are transcribing. This is done via `forced_decoder_ids = processor.get_decoder_prompt_ids(language="Marathi", task="transcribe")` and passing that too the `model.generate` call.

		That's it, let's get transcribing! 🔥
		"""

		# FIXME from peft import PeftModel, PeftConfig
		# FIXME from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

		# FIXME peft_model_id = "reach-vb/whisper-large-v2-hindi-100steps" # Use the same model ID as before.
		# FIXME peft_config = PeftConfig.from_pretrained(peft_model_id)
		# FIXME model = WhisperForConditionalGeneration.from_pretrained(
		# FIXME     peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
		# FIXME )
		# FIXME model = PeftModel.from_pretrained(model, peft_model_id)
		# FIXME model.config.use_cache = True

		# FIXME import gc
		# FIXME import numpy as np
		# FIXME from tqdm import tqdm
		# FIXME from torch.utils.data import DataLoader
		# FIXME from transformers.models.whisper.english_normalizer import BasicTextNormalizer

		# FIXME eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)
		# FIXME forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
		# FIXME normalizer = BasicTextNormalizer()

		# FIXME predictions = []
		# FIXME references = []
		# FIXME normalized_predictions = []
		# FIXME normalized_references = []

		# FIXME model.eval()
		# FIXME for step, batch in enumerate(tqdm(eval_dataloader)):
		# FIXME     with torch.cuda.amp.autocast():
		# FIXME         with torch.no_grad():
		# FIXME             generated_tokens = (
		# FIXME                 model.generate(
		# FIXME                     input_features=batch["input_features"].to("cuda"),
		# FIXME                     forced_decoder_ids=forced_decoder_ids,
		# FIXME                     max_new_tokens=255,
		# FIXME                 )
		# FIXME                 .cpu()
		# FIXME                 .numpy()
		# FIXME             )
		# FIXME             labels = batch["labels"].cpu().numpy()
		# FIXME             labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
		# FIXME             decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		# FIXME             decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
		# FIXME             predictions.extend(decoded_preds)
		# FIXME             references.extend(decoded_labels)
		# FIXME             normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
		# FIXME             normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
		# FIXME         del generated_tokens, labels, batch
		# FIXME     gc.collect()
		# FIXME wer = 100 * metric.compute(predictions=predictions, references=references)
		# FIXME normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
		# FIXME eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

		# FIXME print(f"{wer=} and {normalized_wer=}")
		# FIXME print(eval_metrics)

		"""## Fin!

		If you made it all the way till the end then pat yourself on the back. Looking back, we learned how to train *any* Whisper checkpoint faster, cheaper and with negligible loss in WER.

		With PEFT, you can also go beyond Speech recognition and apply the same set of techniques to other pretrained models as well. Come check it out here: https://github.com/huggingface/peft 🤗

		Don't forget to tweet your results and tag us! [@huggingface](https://twitter.com/huggingface) and [@reach_vb](https://twitter.com/reach_vb) ❤️
		"""