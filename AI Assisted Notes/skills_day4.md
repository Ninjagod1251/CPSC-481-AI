# skills.md — Day 4 module (append to the running file)
*Day 4 = Tue 6/2. Syllabus topic: "Generative AI overview; foundation models." In class this also closed out the **Bayesian Networks** deck (burglar-alarm classwork + exact inference), which is the engine behind Programming Project 1.*

> **Midterm reminder:** 6/11 covers everything 5/26 → 6/10, so *all* of Day 4 is fair game — both the burglar-alarm inference math and the GenAI/LLM concepts.

---

## Skill 3b — The Burglar Alarm Network & Exact Inference by Enumeration
*AIMA §13.2–13.3 · slide deck "Classwork: Burglar Alarm" (Judea Pearl's example)*

### Intuition
Day 3 built the *structure* of a Bayes net (Icy Roads) and the independence story (d-separation). Day 4 adds the **payoff**: given some evidence, *compute a posterior*. The burglar alarm is the canonical example because it has all four "flavors" of node interaction in one tiny graph, and the inference question is one everyone has an intuition for: *the phone rings — was I burgled?*

### The network (AIMA Fig. 13.2)
Causal structure — causes point to effects:

```
   Burglary        Earthquake
        \             /
         v           v
            Alarm
           /     \
          v       v
    JohnCalls   MaryCalls
```

- `Burglary` and `Earthquake` are **root** causes (no parents, just priors).
- Both directly affect `Alarm`.
- `Alarm` is the *only* parent of `JohnCalls` and of `MaryCalls` — the neighbors don't perceive burglaries/earthquakes directly, only the alarm sound. **That single modeling choice is the whole independence story.**

**CPTs (the standard AIMA numbers — verify against the lecture's slide):**

| Node | Probability |
|---|---|
| P(B=t) | 0.001 |
| P(E=t) | 0.002 |
| P(A=t \| B=t, E=t) | 0.95 |
| P(A=t \| B=t, E=f) | 0.94 |
| P(A=t \| B=f, E=t) | 0.29 |
| P(A=t \| B=f, E=f) | 0.001 |
| P(J=t \| A=t) | 0.90 |
| P(J=t \| A=f) | 0.05 |
| P(M=t \| A=t) | 0.70 |
| P(M=t \| A=f) | 0.01 |

Total parameters: 1 + 1 + 4 + 2 + 2 = **10** (vs. 2⁵−1 = 31 for the full joint). That compression is the entire point of a Bayes net.

### The factorized joint (the one formula to memorize)
A Bayes net *is* a factored joint distribution:

> **P(B,E,A,J,M) = P(B) · P(E) · P(A | B,E) · P(J | A) · P(M | A)**

Every inference below is just "sum products of CPT entries from this formula."

### Worked example — P(Burglary | John AND Mary both call)  ✅ verified in code
Query variable `B`; evidence `j, m`; hidden variables `E, A`. Inference by enumeration:

> P(B | j,m) = α · Σ_E Σ_A P(B)·P(E)·P(A|B,E)·P(j|A)·P(m|A)

Expand for B = true (sum 4 terms, each a product of 5 numbers):

| E | A | P(b)·P(e)·P(a\|b,e)·P(j\|a)·P(m\|a) |
|---|---|---|
| t | t | .001·.002·.95·.90·.70 = 0.000001197 |
| t | f | .001·.002·.05·.05·.01 ≈ 0.00000000005 |
| f | t | .001·.998·.94·.90·.70 = 0.000591016 |
| f | f | .001·.998·.06·.05·.01 ≈ 0.00000003 |

Sum (unnormalized P(b,j,m)) = **0.00059224**. Doing the same for B = false gives **0.00149186**.

Normalize: α = 1 / (0.00059224 + 0.00149186), so

> **P(Burglary | j, m) = ⟨0.284, 0.716⟩**

i.e. **~28.4%**. (This is the exact value printed in AIMA — a good self-check that your hand-trace is right.)

### Classwork variant — "Mary calls, John does NOT call"  ✅ verified in code
Same machinery, but now evidence is `¬j, m`, so swap `P(j|A) → P(¬j|A)` (= 1−P(j|A)):

> **P(Burglary | ¬j, m) = ⟨0.0069, 0.9931⟩ → only ~0.69%**

**Why it collapses from 28% to 0.7% — say this in words on the exam:** John almost always calls when the alarm sounds (0.90), so *John staying silent is strong evidence the alarm did NOT go off*. Mary is unreliable (misses it 30% of the time, false-alarms 1%), so her call alone barely moves the needle. The two pieces of evidence pull in opposite directions and "silence from the reliable witness" wins. This is the lesson of the slide: **evidence combines, and an absent observation is still information.**

### The general enumeration recipe (works for ANY Bayes-net query)
1. Name the **query** variable X, the **evidence** e, the **hidden** variables Y.
2. Compute the unnormalized score for *each* value x: `Σ_Y ∏ (CPT entries)`.
3. **Normalize** the vector so it sums to 1 (that's what α does — you never compute α directly, you just divide by the column sum).

### Structural concepts that ride along (likely exam questions)
- **Node ordering matters.** Causes-before-effects → 10 params. A bad order (M, J, E, B, A) → 31 params, *the same as the full joint* — you gained nothing. Any ordering encodes the *same* joint; bad orders just fail to expose independencies.
- **d-separation in this net:** `Burglary ⟂ Earthquake` given ∅ (absolutely independent), but they become **dependent** once `Alarm` is observed — this is **explaining away** (a collider B→A←E). Meanwhile `JohnCalls ⟂ MaryCalls | Alarm` (a fork A→J, A→M, blocked by the observed parent).
- **Markov blanket** of a node = parents + children + children's other parents. E.g. `Burglary`'s blanket is {Alarm, Earthquake}. A node is independent of everything else given its blanket — this is what makes Gibbs sampling local.
- **Complexity:** naive enumeration is O(n·2ⁿ); pushing sums inward (variable elimination) gets it to O(2ⁿ). Exact inference in general Bayes nets is NP-hard, which is why §13.4 introduces sampling.

### 🔗 Direct tie-in to Programming Project 1 (Skill 3c)
`diagnostics.py` is *exactly this algorithm on a different graph*. The "Asia" / chest-clinic network swaps {Burglary, Earthquake, Alarm, John, Mary} for {VisitAsia, Smoking → TB, Cancer, Bronchitis → Xray, Dyspnea}. Your `diagnose(asia, smoking, xray, dyspnea)` does three things and nothing more:
1. build the net (CPTs) in `__init__`,
2. translate the four dropdown strings into an `evidence` dict (skip any `"NA"`),
3. call AIMA's `enumeration_ask('TB'/'Cancer'/'Bronchitis', evidence, net)` and `max()` the three posteriors.
The GUI formats the return `[disease, prob]` as `"{disease} with chance {p*100:.2f}%"`. Reference solution reproduces the spec's **"Cancer with chance 43.67%"**. *You implement zero inference math — the textbook's `enumeration_ask` is the same enumeration you just did by hand above.*

### Common misconceptions
- ❌ "α is a probability I have to compute." → α is just **1 / (sum of the unnormalized scores)**; it's bookkeeping, not a quantity you reason about.
- ❌ "More arrows = more accurate." → Arrows you don't need cost you exponentially many parameters for negligible accuracy. Sparsity is a feature.
- ❌ "B and E are independent, period." → Only **marginally**. Observing their shared effect `Alarm` couples them (explaining away).
- ❌ Forgetting to use P(**¬**j | A) when the evidence is "John did *not* call." The CPT gives P(j=t|A); take the complement.

---

## Skill 4 — Generative AI, Foundation Models & LLMs
*Syllabus Day 4 "Generative AI overview; foundation models" · AIMA Ch. 24 (Language Models) + Ch. 25 (Deep Learning for NLP). The deep LLM/prompting + RAG lectures come 6/8–6/9, but the conceptual spine is set today.*

### Terminology to lock down (the lecture opened here)
| Term | Meaning |
|---|---|
| **AI** | a *field of study* / a set of capabilities |
| **AI system** | a *specific technology*, e.g. ChatGPT |
| **Machine Learning (ML)** | using historic *data* to make predictions; **one approach** to AI (there are non-ML approaches) |
| **Model** | the *"thing"* an ML system produces — feed it inputs, it emits an output (a prediction, or generated text/image) |
| **Generative AI** | models whose *output is new content* (text, image, audio) rather than just a label |

Major AI domains named in the deck: ML · computer vision & speech · NLP · knowledge representation & reasoning · robotics. GenAI/LLMs sit at the intersection of **ML + NLP**.

### Generative vs. discriminative (a recurring exam distinction)
- **Discriminative** model learns **P(class | input)** — draws boundaries, labels things. Lower error, but *can't generate*. (Logistic regression, most classifiers.)
- **Generative** model learns the joint / **P(input)** (or P(input | class)) — *can sample new data*. (Naïve Bayes from Day 2 is the simplest generative model; LLMs are the headline example.)
- Trade-off (AIMA §24.1): discriminative usually wins on accuracy; generative converges faster and needs less data, and — crucially — **can produce novel output**.

### The core idea: a language model is just *next-word prediction*
A language model is a probability distribution over word sequences. By the chain rule:

> **P(w₁…w_N) = ∏ⱼ P(wⱼ | w₁…wⱼ₋₁)**

"Generation" = repeatedly sample the next word given everything so far. Everything from GPT to ChatGPT is a very good estimator of that one conditional. **Tokenization** (splitting text into units — note "aren't" is genuinely ambiguous to split) is the unglamorous first step.

### The road to LLMs (know this progression — it's the narrative of Ch. 24→25)
1. **n-gram models** — approximate the conditional with only the last *n−1* words: P(wⱼ | wⱼ₋ₙ₊₁…wⱼ₋₁). Cheap, but two fatal flaws: data sparsity (most n-grams never seen) and, as n grows, they **memorize/reproduce training text verbatim** instead of generalizing.
2. **Word embeddings** — represent each word as a dense vector instead of an atomic symbol. Similar words land near each other; analogies fall out of vector arithmetic (king − man + woman ≈ queen). Learned unsupervised from raw text (e.g. **GloVe**: dot product of two word vectors ≈ log of their co-occurrence probability).
3. **RNNs / LSTMs** — process a sequence one token at a time, carrying a hidden state. LSTMs add gates so they can *remember* (e.g. subject number — "The athletes … *compete*") across long gaps. Still: nearby-context bias, fixed-size memory, and **sequential** (slow, can't parallelize).
4. **Seq2seq** — encoder RNN reads the source into a final hidden vector; decoder RNN generates the target from it. Breakthrough for machine translation, but the whole input is crushed into *one* fixed vector (a bottleneck).
5. **Attention** — let the decoder look back at *all* encoder hidden states, weighting the relevant ones, instead of relying on a single vector. Removes the bottleneck.
6. **Transformer** — *"Attention Is All You Need"* (Vaswani et al., 2017): drop recurrence entirely, use **self-attention**. This is the architecture every modern LLM is built on.

### Self-attention & the Transformer (the one architecture to know cold)
Each token is projected three ways via learned matrices:
- **Query** q = W_q·x  — "what am I looking for?" (the token attending *from*)
- **Key** k = W_k·x — "what do I offer?" (tokens attended *to*)
- **Value** v = W_v·x — the content actually mixed in

Attention weight of token *i* on token *j*, and the new encoding cᵢ:

> rᵢⱼ = (qᵢ · kⱼ) / √d  →  aᵢⱼ = softmax over j  →  cᵢ = Σⱼ aᵢⱼ · vⱼ

Key points to be able to state:
- **√d scaling** keeps the dot products numerically stable.
- **All positions computed in parallel** (just matrix multiplies) → the speed win over RNNs, and why GPUs matter.
- **Multi-head attention**: split into m independent attention "heads," concatenate — lets different heads specialize (syntax, coreference, …) so important signals don't get averaged away.
- **Positional embeddings**: self-attention is order-agnostic, so you *add* a learned per-position vector to each word embedding to inject word order.
- **Residual connections + feedforward sub-layer** in each block; stack 6+ layers, output of layer i feeds layer i+1.
- **Encoder** (bidirectional, good for classification, → BERT) vs **decoder** (left-to-right masked attention, generates text, → GPT). Full encoder–decoder = original MT transformer.

### Pretraining → transfer learning → fine-tuning = "foundation models"
- Labeling text is expensive; *raw* text is nearly infinite (the internet adds ~100B words/day; Common Crawl, C4).
- **Pretraining**: train a big model on a generic self-supervised objective (predict the masked/next word) over a huge unlabeled corpus → it learns vocabulary, syntax, idiom, world knowledge.
- **Transfer learning / fine-tuning**: adapt that pretrained model to a specific task with a *small* labeled dataset.
- A **foundation model** is exactly this: one large pretrained model that's adapted to many downstream tasks (QA, entailment, summarization, translation). GPT-2/3, BERT, T5 are the textbook examples; ChatGPT is a fine-tuned, instruction-aligned descendant.
- **Contextual embeddings** are why this beats static embeddings: the vector for *"rose"* differs in "the gardener planted a **rose**" vs "the river **rose** five feet" vs "she **rose** to power" — context disambiguates polysemy, which a single fixed vector can't.

### Worked intuition (no math needed, but say it precisely)
- *Why attention beat RNNs:* an RNN at word 57 of 70 has mostly forgotten word 5 (each update overwrites old state); attention can put weight directly on word 5 with no decay, and does it in parallel.
- *Why "foundation":* the same pretrained base supports translation, summarization, QA, etc. — you build *on top of* it rather than training from scratch each time.

### AIMA references
- §24.1 — language models, n-grams, tokenization, generative vs discriminative
- §25.1 — word embeddings; §25.2 — RNNs for NLP; §25.3 — seq2seq + attention; **§25.4 — the Transformer (self-attention, multi-head, positional)**; §25.5 — pretraining & transfer learning, contextual representations (BERT/GPT-2/T5)

### Common misconceptions / exam traps
- ❌ "LLMs *understand* / *retrieve facts* like a database." → They estimate P(next token); fluency ≠ grounded truth. (This motivates the RAG lecture on 6/9.)
- ❌ "Transformers process text left-to-right like RNNs." → Self-attention sees the whole sequence at once; order comes *only* from positional embeddings. (The *decoder* is masked to be left-to-right; the encoder is bidirectional.)
- ❌ "Attention and self-attention are the same." → Plain attention = target attends to source; **self**-attention = a sequence attends to *itself* (source→source, target→target).
- ❌ "Pretraining and fine-tuning are the same step." → Pretrain once on huge unlabeled data; fine-tune cheaply per task on small labeled data.
- ❌ "Naïve Bayes is discriminative." → It's **generative** (models P(features | class)); good Day-2 ↔ Day-4 connection to cite.

---

## Topic map — updated through Day 4

| Day | Date | Topic | AIMA | skills.md |
|---|---|---|---|---|
| 1 | 5/26 | Intro to AI; environments | 1–2 | — |
| 2 | 5/27 | Bayes rule, Naïve Bayes | 12.1–12.6 | Skill 1, 2 |
| 3 | 6/1 | Bayesian networks (Icy Roads, d-sep) | 13.1–13.2 | Skill 3 |
| **4** | **6/2** | **Burglar alarm + exact inference; GenAI/foundation models** | **13.2–13.3; 24–25** | **Skill 3b, 4 (+Project 1 = 3c)** |
| 5 | 6/3 | ML: perceptron, logistic reg, neural nets | 19.1–4, 19.6, 21.1 | (next) |
| — | 6/8 | LLMs: intro, prompting | — | extends Skill 4 |
| — | 6/9 | Retrieval-Augmented Generation (RAG) | — | extends Skill 4 |
| — | 6/10 | Ethics of AI | 27.3 | — |
| — | **6/11** | **MIDTERM (covers 5/26→6/10)** | — | — |

## Session log
- **6/2 (Day 4):** Added burglar-alarm network + exact inference by enumeration (verified P(B\|j,m)=⟨0.284,0.716⟩ in code; derived classwork variant P(B\|¬j,m)≈0.69%). Tied it to Project 1 (`diagnostics.py` = same enumeration on the Asia net). Added the GenAI/foundation-models spine: terminology, generative vs discriminative, LM = next-word prediction, the n-gram→embeddings→RNN→seq2seq→attention→Transformer progression, self-attention (Q/K/V), and pretraining/fine-tuning.
- **Open items:** grad-student ~2-page AI-topic report still to be scoped — *the Transformer / attention or a focused "how RAG mitigates LLM hallucination" angle would map cleanly onto this Day-4 material and the 6/9 lecture.*
