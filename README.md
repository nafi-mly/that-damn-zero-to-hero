# that-damn-zero-to-hero
My personal journey through Karpathy's legendary ["Neural Networks: Zero to Hero"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) playlist.

I lowkey wanna be an inference engineer someday (maybe), which is why I'm starting here with the ultimate "Hello World." Documenting every step even if nobody reads this (at least for now).

## Structure
- `01-micrograd/` - Building backprop from scratch
- `02-bigram/` - Building an MLP language model
- ... (WIP)

> Each subfolder has a `/notes/` directory. Basically my raw paper notes, digitized. It'll probably be cringe to look back on someday, but whatever.

## (Probably) the best way to tackle this playlist
For every single video:

**1. PASS 1 (The Skim):** Copy the video link and just ask AI to brief it. The goal is to get the big picture on what's the goal on that video.

**2. PASS 2 (The Grind):** Watch, pause, and code along **MANUALLY**. Pause and actually ponder what the math represents. This will take 6–10 hours instead of 2. Don't worry, it's normal.

**3. PASS 3 (The Solo Run):** Close everything. Grab a pen and paper. Sketch out the architecture, then **BUILD THAT DAMN THING FROM SCRATCH. NEVER EVER COPY A SINGLE LINE.** Peek at your notes only if you've been stuck for 10+ minutes. Your code will probably look wildly different from the original and 10x uglier than you'd like (that's the entire point). Go check how disgusting my code is in `/from-scratch.ipynb` if you need proof.

Oh, go watch my git history if you want prove that this is a modular step by step tutorial, GLHF ;)

Anyway.... (because finally i get it)

---
# How Large Language Models (LLMs) Work

---

## 1. Architecture Pipeline

### Embedding Layer
* **Word Chunks:** Raw text broken into subword units/tokens.
* **Token ID:** Integer index assigned to each token.
* **Token Embedding:** Mapping token IDs to continuous vector space.
* **Positional Embedding:** Injecting sequence order and positional awareness into the vectors.

---

### Transformer Block (~N times)

#### Self-Attention Sub-Layer
* **RMS Normalization:**
  * Avoids exploding numbers and saves memory.
  * Normalizing ensures all inputs are treated equally.
* **Linear Projection Prep:**
  * **Queries (Q):** *"What am I looking for?"*
  * **Keys (K):** *"What can I offer to other tokens?"*
  * **Values (V):** *"Here is my raw information."*
* **Scaled Dot-Product Attention:**
  `Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_model) + M ) * V`
  * **Causal Masking ($M$):** Upper-triangle filled with `-inf` to prevent token `t` from looking into future tokens.
* **Output Projection:**
  * Mixes multi-head attention outputs back into a single `d_model` vector.
* **Residual Connection:**
  `x = x + Attention(x)`

#### Feed-Forward Network (FFN) Sub-Layer
* **RMS Normalization**
* **Up-Projection:**
  * Expands vector dimension to $4\times$ larger to gain more information space.
* **Activation Function (SwiGLU, GELU, ReLU):**
  * Bends vector directions non-linearly.
* **Down-Projection:**
  * Squeezes vector dimension back to `d_model`.
* **Residual Connection:**
  `x = x + FFN(x)`

---

### Final Layers
* **Final Layer Norm:** Cleans up activations after passing through $N$-stacked Transformer blocks.
* **Linear Head (LM Head):** Projects the `d_model` vector containing previous context across the entire vocabulary space.
* **Softmax:** Calculates probabilities to predict the next token.

---

> **Key Idea:** We define an ideal world of such a beautiful dimension that can predict tokens well, let calculus find it by comparing our results and expectations, then move slowly.

---

## 2. Visual & Intuitive Mental Model

1. **Embedding Space & Order:** We place our words (as token IDs, as vectors, as dots) into a 4096-dimensional space. We use mathematical invariants like vector addition, trigonometric wave frequencies sin/cos, or 2D plane rotations (complex numbers / RoPE) to deliberately bake temporal order into the high-dimensional coordinates
2. **Contextual Interaction (Self-Attention):** We let each vector interact with all other vectors via self-attention. The goal is to shift and rearrange the vectors closer to one another based on previous context.
3. **Feature Expansion & Squeezing (FFN):** The representations are still noisy. We explode the vectors into a $4\times$ higher dimension to separate features clearly, bend them non-linearly to isolate concepts, and then squeeze them back down.
4. **Iterative Refinement:** Repeat steps 2 and 3 through 32 layers, as each layer shifts the vector through deeper logical reasoning space.
5. **Vocabulary Projection:** Project our final vectors onto the full vocabulary list and evaluate which word aligns best using Softmax.
6. **Next Token Generation:** We obtain **a single new token** that is probabilistically the best fit. Append it to the input sequence and feed it back into the machine until it outputs `<EOF>`.