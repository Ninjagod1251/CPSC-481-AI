# CPSC 481 – Artificial Intelligence
## Day 3 Notes: Bayesian Networks
**Date:** June 1, 2026 | **Instructor:** Dr. Anand Panangadan | **AIMA:** Ch. 13.1–13.4

---

## 1. Why Bayesian Networks?

### The Problem
A full joint distribution over **n** boolean variables needs **2ⁿ** entries.
- 10 variables → 1,024 entries
- 30 variables → **1,073,741,824** entries (over a billion)

Impossible to store, estimate from data, or reason about at scale.

### The Solution
Exploit **conditional independence**: most variables don't directly influence most others. A Bayesian Network stores only local dependencies — one small CPT per variable.

---

## 2. Bayesian Network: Formal Definition

A **directed acyclic graph (DAG)** where:
- Each **node** = a random variable
- Each node Xᵢ has a **Conditional Probability Table** P(Xᵢ | Parents(Xᵢ))
- A missing edge = a conditional independence assertion

### The Factorization Formula
$$P(X_1, \dots, X_n) = \prod_i P(X_i \mid \text{Parents}(X_i))$$

This is the single most important formula of the chapter.

### Using a BN to compute a joint entry
Plug values directly into the factorization:
$$P(A=t, B=f, C=f, D=t) = P(+a) \cdot P(\neg b \mid +a) \cdot P(\neg c \mid \neg b) \cdot P(+d \mid \neg b)$$

> **Exam tip:** Read the CPT *row* based on the parent's value, then take complement if the child is false.

---

## 3. Compactness

### CPT Size
A boolean variable Xᵢ with **k** boolean parents has a CPT with **2ᵏ rows** (one per parent combination). Each row stores **1 number** (P(Xᵢ=true | parents); false = 1 minus that).

### Whole Network
If every variable has ≤ k parents and there are n variables:

| Representation | Size |
|---|---|
| Full joint distribution | **O(2ⁿ)** |
| Bayesian network | **O(n · 2ᵏ)** |

**Example — n=30, k=5:**
- Full joint: 2³⁰ ≈ 1 **billion**
- BN: 30 × 32 = **960**

### Why It Works
This is the **local structure principle**: each node interacts directly with only a bounded number of others. Every missing edge = an independence claim = one fewer dimension to enumerate.

> **Key insight:** This same principle underlies Kademlia DHT (O(log n) routing per node), deep neural networks (each neuron only connects to adjacent layers), and Google's consistent hashing systems.

---

## 4. Conditional Independence & D-Separation

### The Three Connection Types

| Pattern | Structure | Blocked when… |
|---|---|---|
| **Chain** | X → Y → Z | Y is observed |
| **Fork** (common cause) | X ← Y → Z | Y is observed |
| **Collider** (common effect) | X → Y ← Z | Y is **NOT** observed (opens when Y observed) |

> **Collider is the opposite of chain/fork.** Observing the common *effect* creates dependence between its causes — this is **explaining away**.

### Icy Roads Example (Fork)
```
    Road Icy?
    /       \
   v         v
A accident  B accident
```
A and B have no arrow between them but are marginally dependent (both caused by road condition).

| Query | Formula | Answer |
|---|---|---|
| P(+a) prior | 0.3(0.2) + 0.1(0.8) | **0.14** |
| P(+a \| +icy) | read CPT directly | **0.30** |
| P(+a \| +b), icy unknown | P(+a,+b)/P(+b) = 0.026/0.14 | **≈ 0.186** |
| P(+a \| +b, ¬icy) | = P(+a\|¬icy); B irrelevant | **0.10** |

**The lesson:** P(A) rises 0.14 → 0.186 when B crashes (B acts as a proxy for icy roads), then drops back to 0.10 once the road condition is observed. Conditioning on the fork's cause **removes** the correlation.

---

## 5. Markov Blanket

**Definition:**
$$MB(X) = \text{Parents}(X) \cup \text{Children}(X) \cup \text{Co-parents of children}(X)$$

A node is conditionally independent of the entire rest of the network given its Markov blanket.

The co-parents group exists because children are **colliders** — once observed, their other parents become relevant (explaining away).

### Classwork: Car-Start Network

**MB(no charging):**
- Parents: {alternator broken, fanbelt broken}
- Children: {battery dead}
- Co-parents of battery dead: {battery age}
- **MB = {alternator broken, fanbelt broken, battery dead, battery age}**

**MB(battery dead):**
- Parents: {battery age, no charging}
- Children: {battery meter, battery flat}
- Co-parents of battery meter: none (single parent)
- Co-parents of battery flat: none (single parent)
- **MB = {battery age, no charging, battery meter, battery flat}**

---

## 6. Exact Inference: Variable Elimination (VE)

### Problem with Enumeration
Naive inference re-computes the same sub-expressions repeatedly. Cost is exponential.

### VE's Key Idea: Dynamic Programming
Compute intermediate results once, cache them as **factors**, reuse them. Like factoring an algebraic expression:

```
a·b + a·c + a·d + a·e·h + a·f·h + a·g·h    → 14 operations
a·(b + c + d + h·(e + f + g))               →  7 operations
```

### Two Operations
1. **Pointwise product** — multiply factors sharing variables into one larger factor
2. **Sum-out (marginalize)** — sum over a hidden variable's values to eliminate it, producing a smaller cached factor

### Example: A → B → C, compute P(C)

Naive: loops over all A and B combinations for every C.

VE pulls the A sum inside:
$$P(C) = \sum_B P(C|B) \underbrace{\sum_A P(A)\,P(B|A)}_{f_A(B)}$$

Compute fₐ(B) once (2 numbers). A is eliminated. Never re-evaluate A again.

### Complexity
- Worst case: **exponential** (size of the largest intermediate factor)
- Practical for sparse/tree-like networks (low **tree-width**)
- On polytrees: **linear** in network size

### Real-World Analogy
Weather variable in a traffic BN: eliminate Rain early → compute its combined effect on all downstream variables (traffic, accidents, visibility) once, cache as one factor. Never re-ask "is it raining?" when considering individual routes. Same principle used in Waze/Google Maps routing.

---

## 7. Approximate Inference: Sampling Methods

### Why Sampling?
Even VE is exponential in the worst case. For large networks: approximate answers that improve with more samples.

**Core idea:** Generate many random "worlds" from the BN's CPTs. Count how often the query is true. Use frequency as probability estimate.

---

### Method 1: Direct (Prior) Sampling
**Algorithm:**
1. Walk variables in topological order (parents before children)
2. At each node, sample from P(Xᵢ | parent values already sampled)
3. Repeat N times; count query frequency

**Problem:** Ignores evidence entirely. Samples from the prior, not the posterior.

---

### Method 2: Rejection Sampling
**Algorithm:** Same as direct sampling, but **discard** any sample that doesn't match the observed evidence.

**Problem:** If P(evidence) is small, nearly all samples are rejected. Extremely wasteful — may need millions of samples to get a useful count.

---

### Method 3: Likelihood Weighting
**Algorithm:**
1. **Fix** evidence variables to their observed values (don't sample them)
2. Sample all non-evidence variables in topological order as normal
3. Each sample gets a **weight** = product of P(evidence_var | its parents) across all evidence variables
4. Count weighted samples; normalize

**Why better:** No samples are wasted — every sample counts (with appropriate weight).

**Problem:** If evidence variables appear late in the network (downstream), upstream variables are sampled with no guidance from the evidence. Most samples get near-zero weight. Estimate degrades.

$$w(\mathbf{z}) = \prod_{i=1}^{m} P(e_i \mid \text{Parents}(E_i))$$

---

### Method 4: Gibbs Sampling (MCMC)
**Algorithm:**
1. Initialize all non-evidence variables randomly
2. Repeat many times:
   - Pick one non-evidence variable Xᵢ
   - Resample Xᵢ from P(Xᵢ | Markov blanket(Xᵢ))
   - Keep everything else fixed
3. Track value counts for query variable; normalize

**Key formula:**
$$P(X_i \mid MB(X_i)) = \alpha \, P(X_i \mid \text{Parents}(X_i)) \prod_{Y \in \text{Children}(X_i)} P(Y \mid \text{Parents}(Y))$$

= multiply the CPT of Xᵢ with the CPTs of all its children.

**Why it works:** After enough steps, the chain reaches the true posterior distribution (requires ergodicity).

**When it fails:** Near-deterministic CPTs (zeros or near-zeros in CPT) partition the state space into isolated regions. Chain gets trapped, never reaches correct posterior. AIMA's Cloudy→Rain example: if P(Rain|¬Cloudy)=0, the chain can never move between Rain=true and Rain=false regions.

### Comparison Table

| Method | Handles Evidence? | Wastes Samples? | Exact? | Best When |
|---|---|---|---|---|
| Direct sampling | ✗ | n/a | No | No evidence |
| Rejection sampling | ✓ | Yes | No | P(e) not too small |
| Likelihood weighting | ✓ | No | No | Evidence upstream |
| Gibbs sampling | ✓ | No | No | Large networks, no zeros in CPT |

> **VE vs Gibbs tradeoff:** For small networks with near-deterministic CPTs (like Cloudy/Sprinkler/Rain/WetGrass where P(W|¬s,¬r)=0), use VE — it's exact, simpler, and doesn't break on zero-probability entries. Gibbs is for large networks where VE's intermediate factors become intractably large.

---

## 8. Classwork Solutions

### Classwork 1: Icy Roads (see Section 4 above)

### Classwork 2: Four-Node Network
**Network:** A → B → C, B → D
**CPTs:** P(+a)=0.4, P(+b|+a)=0.3, P(+b|¬a)=0.99, P(+c|+b)=0.1, P(+c|¬b)=0.6, P(+d|+b)=0.95, P(+d|¬b)=0.98

**Query:** P(A=true, B=false, C=false, D=true)

$$= P(+a) \cdot P(\neg b|+a) \cdot P(\neg c|\neg b) \cdot P(+d|\neg b)$$
$$= 0.4 \times 0.7 \times 0.4 \times 0.98 = \mathbf{0.10976}$$

### Classwork 3: Markov Blankets (see Section 5 above)

---

## 9. Key Connections (Grad-Level Insight)

**Local structure is a universal principle:**
- BN compactness: O(n·2ᵏ) because each variable has ≤k parents
- Kademlia DHT: O(log n) routing because each node has bounded k-buckets
- Deep neural networks: each neuron connects only to adjacent layers
- Google consistent hashing: O(1) lookup via bounded local routing tables

All achieve global expressiveness from bounded local interactions.

**BNs → Modern AI:**
- Naive Bayes (Gmail spam filter) = simplest BN: one class node, all features as children
- Deep learning = BN with learned CPTs (weights), many layers
- Transformers = soft d-separation via attention (each token attends to bounded context)

---

## 10. Midterm Watch List (June 11)

- ✅ BN factorization: write joint as product of CPTs
- ✅ Computing joint entries from a given network
- ✅ Three connection types + when each path is blocked/open
- ✅ D-separation: given evidence, determine if two nodes are independent
- ✅ Markov blanket: identify for any node in a given network
- ✅ Compactness calculation: given n and k, compare BN vs full joint size
- ✅ Variable elimination: trace through one step (factor creation, sum-out)
- ✅ Sampling: identify which method to use given network structure and evidence
- ✅ Gibbs failure condition: when near-zero CPT entries break ergodicity

---

*AIMA Ch. 13.1–13.4 | Next: Generative AI / Foundation Models (June 2)*
