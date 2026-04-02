# Technical Implementation Methodology: Doc-to-LoRA (D2L)

## 1. Goal and Background
[cite_start]Large Language Models (LLMs) experience latency, memory exhaustion, and quality degradation when processing long input sequences due to quadratic attention costs and KV-cache growth[cite: 5, 6, 21, 22]. [cite_start]While Context Distillation (CD) effectively compresses information into model parameters, executing per-prompt CD is too computationally expensive for real-time applications[cite: 7, 27, 30]. 

[cite_start]The Doc-to-LoRA (D2L) approach aims to solve this by meta-learning the context distillation process into a lightweight hypernetwork[cite: 8, 33]. [cite_start]During inference, this hypernetwork performs a single forward pass over a context to generate a context-specific LoRA adapter, instantly internalizing the information so the LLM can answer subsequent queries without reading the context again[cite: 8, 9, 35].

---

## 2. Architecture Design

The D2L architecture acts as a bridge between the base LLM's token activations and its target LoRA weight parameters.

### 2.1. Base LLM Setup
* [cite_start]**Base Model:** The primary implementation utilizes `gemma-2-2b-it`[cite: 201].
* [cite_start]**Target Modules:** Generated LoRA matrices are applied exclusively to the "down projection" layer of each MLP block within the base LLM[cite: 243, 253].
* [cite_start]**Input Extraction:** For each layer l, the hypernetwork consumes the frozen LLM's per-layer token activations (Z_{l-1}) with shape (N x D), where N is the number of context tokens and D is the hidden size[cite: 152, 153].

### 2.2. The D2L Hypernetwork
* [cite_start]**Module Type:** The hypernetwork is a Perceiver-style cross-attention module[cite: 158]. 
* [cite_start]**Depth:** It consists of 8 cross-attention blocks[cite: 241]. [cite_start]It explicitly contains no self-attention layers[cite: 241].
* [cite_start]**Total Parameters:** Approximately 309M trainable parameters[cite: 244].
* [cite_start]**Latent Query Formulation:** The model utilizes learnable, input-independent latent queries (Q_m) with shape (r x d_q), where r is the target LoRA rank[cite: 160].
* [cite_start]**Heads:** Cross-attention between Q_m and Z_{l-1} yields latent vectors, which are then passed through two per-layer output heads[cite: 161, 162, 163]. [cite_start]These heads map the latents into the target LoRA matrices A_l and B_l[cite: 163].

### 2.3. Chunking Mechanism (Handling Long Contexts)
To process contexts longer than the training sequences without changing the hypernetwork's output shape, D2L employs a chunking mechanism:
* [cite_start]**Partitioning:** The input context is partitioned into K contiguous chunks[cite: 171].
* [cite_start]**Independent Processing:** Each chunk is passed through the hypernetwork independently, producing per-chunk adapters A_l^(k) and B_l^(k)[cite: 171].
* [cite_start]**Composition:** The chunks are combined by concatenating along the rank dimension[cite: 172]. [cite_start]Matrix A_l is formed by vertically stacking A_l^(1) through A_l^(K), and B_l is formed by horizontally concatenating B_l^(1) through B_l^(K)[cite: 175, 189].
* [cite_start]**Effective Rank:** If the base rank is r, the final adapter for K chunks has an expanded rank of r * K[cite: 190].
* [cite_start]**Chunk Sizes:** During the main QA evaluations, equal-sized chunks of 8K tokens are used[cite: 242].

---

## 3. Meta-Training Data Generation

Training the hypernetwork requires a large, diverse dataset containing contexts, queries, and target distributions.

### 3.1. Context Corpus
* [cite_start]**Sources:** Use a ~900 million token subset of FineWeb-Edu, combined with passages from SQuAD, ROPES, and DROP[cite: 672, 673].
* [cite_start]**Filtering:** Filter out any passages longer than 10,000 characters to yield ~3.2 million unique contexts[cite: 674].

### 3.2. Query Generation
* [cite_start]**Teacher Model:** Use a larger instruct model (e.g., `gemma-3-12b-it`) to generate 10 context-grounded queries per sample[cite: 675].
* [cite_start]**Iterative Generation:** Prompt the model in two iterations of 5 queries each[cite: 678]. [cite_start]For the second iteration, inject the first 5 generated query-answer pairs into the prompt as in-context examples to force the generation of non-overlapping, increasingly complex queries[cite: 679].
* [cite_start]**Discard Answers:** Discard the generated answers; only keep the queries[cite: 680].
* [cite_start]**Augmentation:** Randomly place generated queries into various instruction templates to increase prompt diversity[cite: 681].

### 3.3. Target Logit Extraction
* [cite_start]**Process:** For every unique context-query pair, feed it into the frozen target LLM (`gemma-2-2b-it`) to generate a single response[cite: 682].
* [cite_start]**Target Data:** Record the top-16 token logit values for every generated token[cite: 682]. [cite_start]This dense logit distribution serves as the distillation target[cite: 683].

---

## 4. Training Methodology

### 4.1. Loss Objective
[cite_start]D2L optimizes a query-independent Context Distillation (CD) loss[cite: 127, 138]. 
* [cite_start]Do not use standard Next-Token Prediction (NTP); empirical results show that NTP causes severe drops in recall[cite: 723, 728, 729]. 
* [cite_start]Instead, minimize the Kullback-Leibler (KL) divergence between the context-conditioned teacher distribution and the context-internalized student distribution[cite: 130, 145, 146, 733].

### 4.2. Two-Stage Training Protocol
[cite_start]To prevent training instability regarding compositional LoRAs, implement a two-stage curriculum[cite: 694]:
1.  [cite_start]**Stage 1 (Pure Internalization):** Train D2L to output exactly one chunk for every input context for 80,000 gradient steps[cite: 695].
2.  [cite_start]**Stage 2 (Composition Regularization):** Train for an additional 20,000 steps using randomized chunking[cite: 698]. For each input, randomly assign it to:
    * [cite_start]1 chunk (50% probability) [cite: 697]
    * [cite_start]2 chunks (12% probability) [cite: 697]
    * [cite_start]3-8 chunks (37.5% probability, distributed equally) [cite: 697]

### 4.3. Hyperparameters and Optimization
* [cite_start]**Context Length Range:** Training samples typically range from 32 to 256 tokens [cite: 611][cite_start], with a maximum sequence length of ~2344 tokens depending on the exact corpus sample[cite: 315, 640].
* [cite_start]**LoRA Target Rank:** Base rank r = 8 per chunk[cite: 242].
* [cite_start]**Batching Strategy:** Pack context inputs into dense 4K-token sequences[cite: 699].
* [cite_start]**Effective Batch Size:** Use heavy gradient accumulation to achieve an effective batch size exceeding 200,000 context tokens per step (crucial for convergence)[cite: 699, 700].
* [cite_start]**Learning Rate:** 4e-5[cite: 612].

---

## 5. Inference Execution

When deploying the trained D2L model to internalize an unseen document:
1.  [cite_start]**Chunking:** Split the target document into 8K-token chunks[cite: 242].
2.  [cite_start]**Activation Extraction:** Run a single forward pass of the document through the frozen LLM to obtain hidden states[cite: 152].
3.  [cite_start]**Hypernetwork Generation:** Pass the activations through the D2L Perceiver module[cite: 154]. [cite_start]This can be executed in "batched mode" (computing all layers simultaneously for maximum speed) or "iterative mode" (layer-by-layer to minimize peak VRAM)[cite: 245, 246].
4.  [cite_start]**Adapter Composition:** Concatenate the resulting A and B matrices across chunks to build the final high-rank LoRA adapter[cite: 171, 172].
5.  [cite_start]**LLM Inference:** Inject the LoRA weights into the LLM's MLP down-projection layers[cite: 243]. [cite_start]You can now serve any number of downstream queries without including the original document in the KV-cache[cite: 36, 239].