# GPT 2

## About

This is the next iteration of gpt. Usually, transformer models have an encoder and a decoder. But GPT-2 is a decoder only model, therefore, the cross-attention block that uses the encoder is also missing.

The major difference between this model and GPT-1 is the shuffeling of layer normalisations, and the addition of additional layer normalisation to the final self-attention block.

The model has 124 million parameters and is supposed to be better performing than the OpenAI GPT-2 model that was released in 2019.

Since this is a fairly more complicated model than GPT-1, we will be doing the code explanations in a separate EXPLANATION.md file. The python file will contain only necessary comments related to the code.
