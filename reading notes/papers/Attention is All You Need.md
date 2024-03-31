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
3. **Attention**
    * **Attention function**: mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
    * **Output**: computed as a _weighted sum_ of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
    * **Scaled Dot-Product Attention**:
        * The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.
        * We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
        * We Compute the matrix of output as:

            $$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

            * $Q$: query matrix
            * $K$: key matrix
            * $V$: value matrix
        * Two most commonly used attention functions:
            * **Additive attention**
            * **Dot-product (multiplicative) attention**
    * **Multi-Head Attention**:
        * Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values and queries, we found it beneficial to _**linearly project the queries**, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively_.
        * On each of these projected versions of queries, keys and values, we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values.
        * Multi-head attention allows the model to focus on different parts of the sequence. With a single attention head, averaging inhibits this.

            $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \\ \text{where } \text{head}_i = \text{Attention}(QW_i^Q , KW_i^K , VW_i^V)$$

            * Projections:
                * $W_i^Q \in \R^{d_{model} \times d_k}$
                * $W_i^K \in \R^{d_{model} \times d_k}$
                * $W_i^V \in \R^{d_{model} \times d_v}$
        * In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{\text{model}}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.
    * **Applications of Attention in our model**:
        * _In **"encoder-decoder attention" layers**, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder._ This allows every position in the decoder to _attend over all positions in the input sequence_. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models.
        * The **encoder** contains **self-attention layers**. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can _attend to all positions in the previous layer of the encoder_.
        * Similarly, **self-attention layers** in the **decoder** allow each position in the decoder to _attend to all positions in the decoder up to and including that position_. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.
4. **Position-wise Feed-Forward Networks**
    * In addition to attention sub-layers, each of the layers in our encoder and decoder contains a _fully connected feed-forward network_, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

        $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
    * While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_\text{model} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$.
5. **Embeddings and Softmax**
    * We use _learned embeddings_ to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$.
    * We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
    * In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation.
    * In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.
6. **Positional Encoding**
    * Since our model contains _no recurrence_ and _no convolution_, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.
    * To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed.
        * There are many choices of positional encodings, learned and fixed. In this work, we use sine and cosine functions of different frequencies.

            $$\text{PE}(pos, 2i) = \sin(pos/10000^{2i/d_{\text{model}}}) \\ \text{PE}(pos, 2i+1) = \cos(pos/10000^{2i/d_{\text{model}}})$$

![Transformer model architecture](../images/Attention%20is%20All%20You%20Need/transformer%20architecture.png)

![Scaled dot-product attention](../images/Attention%20is%20All%20You%20Need/scaled%20dot-product%20attention.png)
![Multi-head attention](../images/Attention%20is%20All%20You%20Need/multi-head%20attention.png)

### Complexities of Different Layer Types

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|------------|----------------------|-----------------------|---------------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n / r)$ |

## Why Self-Attention

Self-attention advantages:

1. Total computational complexity per layer.
2. The amount of computation that can be parallelized, as measured by the _minimum number of sequential operations required_.
3. The path length between long-range dependencies in the network: The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.