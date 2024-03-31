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

1. **Encoder-Decoder Architecture**
    * **Encoder**: The encoder maps an input sequence of symbol representations $(x_1, \dots, x_n)$ to a sequence of continuous representations $z = (z_1, \dots, z_n)$.
    * **Decoder**: Given $z$, the decoder then generates an output sequence $(y_1, \dots, y_m)$ of symbols one element at a time.
    * At each step the model is _auto-regressive_, consuming the previously generated symbols as additional input when generating the next.
2. **Encoder and Decoder Stacks**
    * **Encoder**: Has a stack of $N = 6$ identical layers.
        * Each layer has two sub-layers:
            1. a multi-head self-attention mechanism;
            2. a simple, position-wise fully connected feed-forward network.
        * Output of each sub-layer: $\text{LayerNorm}(x + \text{Sublayer}(x))$
        * To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_\text{model} = 512$.
    * **Decoder**: Also has a stack of $N = 6$ identical layers.
        * In addition to the two sub-layers in each encoder layer, the decoder inserts a _third sub-layer_, which performs multi-head attention over the output of the encoder stack.
        * Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
        * This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

![Transformer model architecture](../images/Attention%20is%20All%20You%20Need/transformer%20architecture.png)

![Scaled dot-product attention](../images/Attention%20is%20All%20You%20Need/scaled%20dot-product%20attention.png)
![Multi-head attention](../images/Attention%20is%20All%20You%20Need/multi-head%20attention.png)

## Why Self-Attention

Self-attention advantages:

1. Total computational complexity per layer.
2. The amount of computation that can be parallelized, as measured by the _minimum number of sequential operations required_.
3. The path length between long-range dependencies in the network: The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.