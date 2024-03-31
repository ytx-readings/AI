# Attention is All You Need

> [**Download paper**](../../papers/transformer%20model/Attention%20is%20All%20You%20Need.pdf)
>
> **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
>
> [**Code repository**](https://github.com/tensorflow/tensor2tensor)

## Backgrounds

* **Self-attention**: an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
    * Has been used successfully in a variety of tasks, including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.
* **Transformer**: the first transduction model relying entirely on self-attention to compute representations of its input and output _without using sequence-aligned RNNs or convolution_.

## Model Architecture

![Transformer model architecture](../images/Attention%20is%20All%20You%20Need/transformer%20architecture.png)

![Scaled dot-product attention](../images/Attention%20is%20All%20You%20Need/scaled%20dot-product%20attention.png)
![Multi-head attention](../images/Attention%20is%20All%20You%20Need/multi-head%20attention.png)

## Why Self-Attention

Self-attention advantages:

1. Total computational complexity per layer.
2. The amount of computation that can be parallelized, as measured by the _minimum number of sequential operations required_.
3. The path length between long-range dependencies in the network: The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.