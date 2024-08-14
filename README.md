# Documentation for the Fine Tuning Gemma2 model for conversational AI

## Overview

This code is designed to train a language model using the `unsloth` library, specifically for summarizing conversations in Hindi. It leverages the `FastLanguageModel` class to load a pre-trained model, fine-tune it on a specified dataset, and evaluate its performance.

## Requirements

To run this code, ensure you have the following packages installed:

- `unsloth`
- `torch`
- `trl`
- `transformers`
- `datasets`

You can install the necessary packages using the following commands:

```bash
pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -qqq --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## Code Structure

### 1. Importing Libraries

The code begins by importing the necessary libraries and modules, including `FastLanguageModel` from `unsloth`, `torch`, and `datasets`.

### 2. Model Configuration

- **Model Parameters**: The model is configured with a maximum sequence length of 2048 and is set to load in 4-bit precision.
- **Model Name**: The pre-trained model used is `unsloth/gemma-2-9b`.

### 3. Loading the Model

The model and tokenizer are loaded using the `FastLanguageModel.from_pretrained()` method, which takes several parameters such as model name, maximum sequence length, and token.

### 4. Model Fine-Tuning

The model is fine-tuned using the `get_peft_model()` method, which applies parameter-efficient fine-tuning (PEFT) techniques. Key parameters include:

- `r`: Rank of the low-rank adaptation.
- `target_modules`: List of modules to be fine-tuned.
- `lora_alpha`, `lora_dropout`: Parameters for LoRA (Low-Rank Adaptation).

### 5. Preparing the Dataset

The dataset for training is loaded using the `load_dataset()` function. The dataset is then processed to format the prompts for the model using the `formatting_prompts_func()` function.

### 6. Training the Model

The `SFTTrainer` class from `trl` is used to train the model. Key training arguments include:

- `per_device_train_batch_size`: Batch size for training.
- `max_steps`: Total training steps.
- `learning_rate`: Learning rate for the optimizer.

### 7. Monitoring GPU Memory

The code includes functionality to monitor GPU memory usage before, during, and after training, providing insights into the performance and resource utilization.

### 8. Inference

After training, the model is prepared for inference using `FastLanguageModel.for_inference()`. The inference process involves generating summaries based on input articles.

### 9. Saving the Model

Finally, the trained model and tokenizer are saved locally using the `save_pretrained()` method.

## Usage

To use this code, follow these steps:

1. Ensure all dependencies are installed.
2. Configure the model parameters as needed.
3. Run the code to train the model on the specified dataset.
4. Use the trained model for inference on new input data.
5. Save the model for future use.

## Conclusion

This code provides a comprehensive framework for training a language model tailored for summarization tasks in Hindi. It demonstrates the integration of various libraries and techniques to achieve efficient model training and inference.

Citations:
[1] https://github.com/unsloth
