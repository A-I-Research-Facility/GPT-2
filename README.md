# GPT 2

## Requirements

This model should be run on Linux environment to take full advantage of parallel processing that can be done on multi GPUs. Other hardware optimizations do not work on windows or macOS as well. The recommended scenario to run this model is to get a cloud GPU cluster of A100 GPUs as the hardware optimizations I have done are specefically for Ampere architecture.

Also, Batch Size(B) can be altered for A100 series GPUs to 16 instead of 8 that is in the code. I have also ran this code on RTX 3070 just to ensure that this works, but even with 24 GBs of GPU memory, I was not able to exceed 8 as the batch size. For cloud GPU clusters, try 16 or even 32 as the batch size for high end GPUs like NVIDIA H100.

## About

This is the next iteration of gpt. Usually, transformer models have an encoder and a decoder. But GPT-2 is a decoder only model, therefore, the cross-attention block that uses the encoder is also missing.

The major difference between this model and GPT-1 is the shuffeling of layer normalisations, and the addition of additional layer normalisation to the final self-attention block.

The model has 124 million parameters and is supposed to be better performing than the OpenAI GPT-2 model that was released in 2019.

Since this is a fairly more complicated model than GPT-1, we will be doing the code explanations in a separate EXPLANATION.md file. The python file will contain only necessary comments related to the code.
