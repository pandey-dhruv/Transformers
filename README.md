# Transformers
A French - English Transformer based Neural Machine Translation Model

# Overview
The translation model is based upon the original Transformer architecture using and encoder and a decoder. The model is trained on the OpusBooks corpus available on HuggingFace.
Here, we have improved the original model architecture using Rotarty Positional Embeddings that have now become the staple for positional embeddings in the latest Large Language Models
as they are better able to capture language nuances. We also implement Key-Value cache (first implemented in LlaMA, Meta's LLM) to improve model inference speed by 1.5x.

# Installtion Requirements - 
Python 3.11
PyTorch 2.14
tokenizer
tqdm
numpy
pandas

# Inference - 
First ensure that the model has been copied to the GPU VRAM (if planning on using GPU Hardware acceleration, set device='cuda')
Then simply run the inference notebook that has been provided and it will provide the English translations for the first 10 French sentences.

# Feel free to play around with the code :)
