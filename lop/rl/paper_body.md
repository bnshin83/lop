<!--
ICML 2025 Paper Body (Markdown)

This file contains your paper content starting from Introduction.
The title, authors, and abstract are in paper.tex (the LaTeX wrapper).

To convert to LaTeX:
  pandoc paper_body.md --from markdown+tex_math_single_backslash --to latex --natbib --wrap=none -o paper_body.tex

Then compile:
  pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
-->

## Introduction

The goal of continual learning is to develop algorithms that can learn from non-stationary data distributions without losing the ability to acquire new knowledge [@Ring1994; @Thrun1998]. 
While modern neural networks are capable of learning from a fixed dataset and generalizing to unseen data from the same distribution, they struggle to adapt when the data distribution changes over time---a fundamental limitation for real-world deployment where environments evolve continuously [@Lyle2023; @Dohare2023; @Abbas2023; @Nikishin2022].

Two interrelated phenomena characterize this failure mode. **Loss of plasticity** refers to the progressive degradation in a network's ability to learn from new data, even when that data comes from simple, learnable distributions [@Berariu2021; @Lyle2023; @Abbas2023]. 
**Catastrophic forgetting** describes the abrupt loss of previously learned knowledge when adapting to new tasks [@Goodfellow2013; @Kirkpatrick2017]. While forgetting has received substantial attention through methods like Elastic Weight Consolidation (EWC) [@Kirkpatrick2017], Synaptic Intelligence [@Zenke2017], and Memory Aware Synapses [@Aljundi2018], the deeper problem of plasticity loss has only recently gained focused attention.

Plasticity loss is also a central bottleneck in deep reinforcement learning (RL), where non-stationarity is not an edge case but an inherent property of learning: 
as policies improve, state visitation changes, and (for value-based methods) bootstrapped targets drift over time. 
Recent work emphasizes that "plasticity loss" bundles multiple phenomena and must be interpreted through the evaluation setup and the kind of non-stationarity: **input shifts** (changes in \(P(x)\)) versus **target shifts** (changes in \(P(y\mid x)\)), and **trainability** versus **generalizability** (e.g., warm-start "generalization gaps" despite comparable training loss) [@Klein2024Survey; @Berariu2021]. 
This lens matters for method design: some interventions primarily restore trainability (e.g., reviving dormant capacity) [@Sokar2023; @Nikishin2023; @Lyle2023], while others act by controlling optimization/regularization dynamics and can affect generalization in different ways [@Kumar2024; @Klein2024Survey].

Moreover, evidence from on-policy RL suggests that interventions successful in supervised learning or off-policy RL do not always transfer: under several forms of environment-level distribution shift, methods like plasticity injection and CReLU can fail—sometimes even underperforming the no-intervention baseline [@JulianiAsh2024OnPolicy]. In contrast, *continuous* regularization-based methods (L2, LayerNorm, regenerative regularization) can be consistently beneficial, aligning with a broader "bitter lesson" that general network regularizers tend to outperform domain-specific plasticity interventions [@Klein2024Survey; @JulianiAsh2024OnPolicy]. This motivates plasticity-preserving methods that operate continuously and do not rely on replay buffers or explicit shift detection.

While this paper does **not** present reinforcement-learning experiments, we use RL as a motivating and stress-testing lens: many mechanisms and mitigations discussed in the deep RL plasticity literature (non-stationarity type, “where” plasticity is lost in the network, and the distinction between restoring trainability vs. preserving generalizability) directly inform how we design and interpret utility-gated perturbations in the simpler supervised streaming setting [@Klein2024Survey; @JulianiAsh2024OnPolicy].

Recent investigations have uncovered multiple mechanisms underlying plasticity loss. @Sokar2023 and @Dohare2024 identified the **dormant neuron phenomenon**, where an increasing fraction of neurons become permanently inactive during training—a pathology pervasive in both reinforcement learning and continual supervised learning. 
@Kumar2023, @Dohare2024, and @Lyle2023 identified **unbounded parameter norm growth** as a critical failure mode, showing that expanding weights hinder optimization dynamics. @Kumar2021 and @Lyle2023 connected plasticity loss to **decreasing representation rank**, suggesting that networks lose expressivity over time. Most recently, @Lewandowski2024 proposed that **loss of curvature directions** in the Hessian provides a consistent explanation across different settings, showing that previous explanations fail to generalize.

Despite this progress in understanding mechanisms, existing solutions face practical limitations. Regularization-based approaches like weight decay and L2 regularization toward initialization [@Kumar2023] can preserve plasticity but **indiscriminately constrain parameter movement**, potentially preventing **optimal adaptation** needed for new tasks. The **Shrink \& Perturb (S\&P)** method [@Ash2020] injects Gaussian noise to sustain exploration, but **perturbs all parameters uniformly** regardless of their **importance**. Continual Backprop [@Dohare2021; @Dohare2023] tracks neuron-level utility and periodically reinitializes low-utility neurons, but operates at the neuron rather than parameter level and requires discrete reset decisions.

This work presents a systematic empirical study of **Utility-based Perturbed Gradient Descent with weight decay (UPGD-W)** [@Elsayed2024], an optimizer that uses a **scalable utility approximation** to estimate per-parameter **utility** and **selectively gate noise injection into parameter updates**.
 However, the mechanistic basis for explicitly gating noise based on utility remains under-explored. In this paper, we characterize **why** utility gating is effective, revealing that high-utility parameters are **sparse** and **concentrated in the output layer**. Leveraging this understanding, we propose a refined intervention: **precise gating** that confines noise injection to a **tiny fraction** of the output parameters. We theoretically justify this design through two propositions, showing that minimizing the "perturbation budget" while targeting the output layer maximizes plasticity with minimal stability cost.

**Contributions.** We make the following contributions:

1. **Precise output-layer gating:** We discover that confining utility-gated perturbations to the **output layer alone**—modifying a **tiny fraction** of total parameters—consistently outperforms full-network methods, achieving up to 146\% improvement over S\&P. This finding challenges the "more is better" assumption in plasticity injection.

2. **Theoretical justification:** We derive **two propositions** that justify this strategy, showing that concentrating the "perturbation budget" in the output layer maximizes plastic adaptation while minimizing stability costs in the representation.

3. **Utility dynamics insights:** We demonstrate that the **natural sparsity** ("utility tail") emerging from gradient dynamics is essential for effective selection; constraining utilities to narrow ranges degrades performance by up to 43\%, validating the need for distribution-aware gating.

4. **Mechanistic validation:** Through comprehensive evaluation of 8 method variants across 4 benchmarks, we show that our findings align with the curvature-based theory of @Lewandowski2024, offering a practical $O(N)$ implementation that preserves curvature directions without explicit Hessian computation.

## Related Work

### Loss of Plasticity: Mechanisms and Measurements

The phenomenon of plasticity loss has been characterized from multiple perspectives. @Ash2020 first observed that warm-started networks can exhibit a 'generalization gap'—performing worse than randomly initialized ones even on identical data. @Berariu2021 extended this observation, hypothesizing that pretrained networks converge to sharper minima due to reduced gradient noise.

Several mechanistic explanations have been proposed. @Lyle2021 first observed capacity loss in deep RL---the tendency for networks to lose their ability to fit new target functions over time. @Sokar2023 later explicitly identified and named the **dormant neuron phenomenon**---units that become permanently inactive---as a key mechanism driving this capacity loss, a pathology pervasive in both reinforcement learning and continual supervised learning. @Dohare2024 further validated this phenomenon, 
while @Abbas2023 showed that non-saturating activation functions like CReLU can mitigate related activation sparsity issues. 
@Kumar2023, @Dohare2024, and @Lyle2023 identified **unbounded parameter norm growth** as a critical failure mode, while @Nikishin2022 observed that large norms correlate with "primacy bias" in deep RL.

@Kumar2021 and @Lyle2021 used feature rank as a proxy measure for plasticity, suggesting that networks progressively lose expressive capacity. However, @Gulcehre2022 later questioned this relationship by demonstrating weak correlation between feature rank and agent performance. @Lewandowski2024 further demonstrated that these mechanistic explanations are inconsistent---counterexamples exist for each proposed mechanism. They instead proposed that **loss of curvature directions** (measured as reduction in Hessian rank) provides a consistent explanation; 
@He2025 extended this view with the concept of **Hessian spectral collapse**.

### Plasticity Loss in Deep Reinforcement Learning

Deep RL provides a particularly demanding setting for plasticity-preserving optimization: even in a fixed environment, learning is intrinsically non-stationary due to (i) policy-induced changes in the data distribution and, (ii) for value-based methods, drifting bootstrap targets.
 
A recent survey organizes the literature by (i) **definitions/measurement setups** for plasticity loss, (ii) **proposed causes** spanning unit saturation/dormancy, effective-rank collapse, gradient pathologies, curvature/sharpness, parameter norm growth, and replay-ratio effects, and (iii) **mitigations** such as non-targeted/targeted resets, parameter and representation regularization, activation-function choices, and hybrid combinations [@Klein2024Survey]. 
This taxonomy helps clarify which mechanisms a given intervention plausibly addresses (e.g., reducing norm growth vs. reviving dormant capacity) and highlights that different RL regimes (on-policy vs. off-policy; regression vs. classification losses) can exhibit different dominant failure modes.

Critically, evidence from on-policy RL suggests that portability of mitigation techniques is not guaranteed: under several forms of environment-level distribution shift, methods that help in supervised learning or off-policy RL may be ineffective or even harmful, whereas a class of *regenerative* (regularization-like) methods can be consistently beneficial [@JulianiAsh2024OnPolicy]. 
These findings motivate the search for methods that operate *continuously* during training (avoiding discrete intervention points and replay-dependent recovery) while addressing the regime-specific limitations of existing mitigations. The regenerative (regularization-based) methods highlighted by @JulianiAsh2024OnPolicy achieve this through weight-norm regularization toward initialization. In this work, we investigate whether **utility-gated perturbations**—which also operate continuously but through selective noise injection rather than norm regularization—can provide a complementary mechanism for plasticity preservation.

From the perspective of evaluation, the survey also highlights two practical distinctions that are easy to miss when comparing results across papers. First, “plasticity” can refer to adaptation under **input non-stationarity** (requiring representation relearning) versus **target non-stationarity** (often solvable via head adaptation), motivating the widespread use of input-permuted vs. label-permuted benchmarks as complementary stress tests [@Klein2024Survey; @Lyle2023]. Second, improved *trainability* does not guarantee improved *generalizability*: some interventions can restore the ability to fit changing targets while still yielding warm-start generalization gaps or brittle representations [@Klein2024Survey; @Berariu2021]. We use these distinctions below to interpret why layer-selective utility gating behaves differently across input- vs. label-shift benchmarks.

Given these RL-specific confounds—exploration, replay, and bootstrap targets—supervised continual learning benchmarks provide useful *controlled surrogates* for isolating input-shift vs. target-shift adaptation and measuring where plasticity is preserved (e.g., output head vs. representation). We adopt this approach in the experiments below, with translating findings back into RL as a natural but separate line of work.

### Perturbation-Based Methods

Perturbation-based approaches attempt to maintain plasticity by injecting noise during training. The widely-used **Shrink \& Perturb (S\&P)** method [@Ash2020] showed that shrinking weights toward zero and adding Gaussian noise can overcome the warm-start generalization gap. In their original formulation, shrink-perturb is applied at training round boundaries (when new data arrives):
\[
\theta \leftarrow \lambda \theta + \sigma_{\text{sp}} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
\]
with shrinkage factor $\lambda < 1$ and noise scale $\sigma_{\text{sp}}$. Later work adapted this to a *continuous* (per-step) variant integrated with SGD [@Dohare2024; @JulianiAsh2024OnPolicy]. The mechanism balances gradient contributions between old and new data by increasing the loss (and hence gradient magnitude) for previously seen samples.

However, S\&P applies identical perturbation to all parameters, which can disrupt important learned representations. @Dohare2021 introduced **Continual Backprop**, which tracks neuron-level "contribution" based on the product of feature activation and outgoing weight magnitudes. Low-contribution neurons are periodically reinitialized, selectively targeting unused capacity. @Dohare2024 extended this to deep networks, showing that maintaining plasticity through selective reinitialization enables long-horizon learning.

### Parameter Importance and Regularization

A parallel line of work estimates parameter importance for selective regularization. 
**Elastic Weight Consolidation (EWC)** [@Kirkpatrick2017] uses the Fisher information matrix to identify important parameters and adds a quadratic penalty preventing deviation from previous task solutions. 
**Synaptic Intelligence** [@Zenke2017] computes online importance based on the contribution to loss reduction during training. **Memory Aware Synapses (MAS)** [@Aljundi2018] estimates importance using gradients of the network output.

These methods were primarily designed for catastrophic forgetting rather than plasticity loss, and typically assume discrete task boundaries. @Kumar2023 proposed **regenerative regularization**, which regularizes toward initial parameter values rather than zero, showing that this "L2 Init" approach consistently mitigates plasticity loss across settings. The Wasserstein regularizer of @Lewandowski2024 takes this further by regularizing the distribution of weights rather than individual values, allowing greater parameter movement while preserving beneficial initialization properties.

### Layer-wise Learning Dynamics

A growing body of work examines how learning and adaptation are distributed across network layers. Early transfer learning research established that shallow layers learn general, transferable features while deeper layers specialize to task-specific content [@Yosinski2014]. This asymmetry has been repeatedly confirmed in continual learning: @Wei2024 showed that under target label changes, most adaptation occurs in the final classification layer rather than the representation, suggesting that "head adaptation" often suffices under target non-stationarity.

Linear probing studies further support this observation. When fine-tuning pretrained models, retraining only the final layer often captures most performance gains, implying that the output head carries disproportionate task-specific signal. @Anagnostidis2023 demonstrated that memorization—and by extension, task-specific learning—concentrates in late network modules, with the output layer serving as the primary locus of adaptation.

These findings have important implications for plasticity preservation. If adaptation under target shift is head-centric, then interventions applied uniformly across all layers may be inefficient or even harmful to general-purpose representations. This motivates investigation of *layer-selective* approaches: rather than applying noise injection throughout the network, confining perturbation to the output layer—where adaptation pressure concentrates—may achieve better plasticity-stability trade-offs.

### Utility-based Perturbed Gradient Descent

UPGD [@Elsayed2024] synthesizes insights from perturbation and importance-based methods. Rather than applying uniform noise or discrete neuron resets, UPGD **selectively gates noise injection into parameter updates** using a continuous, per-parameter utility score. Parameters with high utility (those contributing to loss reduction) receive smaller updates, while low-utility parameters receive larger, more exploratory updates. The variant with weight decay, **UPGD-W**, incorporates shrinkage toward zero similar to S\&P, combining utility-gated perturbation with weight regularization.


The key innovation is the **scalable utility approximation**: $\tilde{u}_{i} = -g_{i} \cdot \theta_{i}$, which approximates the parameter's contribution to loss reduction without requiring Hessian computation. This is related to the "contribution" measure of @Dohare2021 but operates at the parameter level and modulates continuous updates rather than discrete resets. While empirically effective, the mechanistic basis for *why* utility gating succeeds—and *where* it matters most—remains under-explored. This gap motivates our investigation.

### Online Learning and Dynamic Regret

Our theoretical analysis draws on the online convex optimization framework [@Zinkevich2003]. In online learning, a learner makes sequential decisions and performance is measured by **regret**—the gap between cumulative loss and a comparator. While *static regret* compares against a single fixed optimum, **dynamic regret** [@Besbes2015] compares against a sequence of time-varying optimal solutions, appropriate for non-stationary environments where the optimum drifts over time.

A key result from this literature is that dynamic regret scales with the **path length** of the comparator sequence: $\sum_{t} \|w^*_{t+1} - w^*_{t}\|$. When the optimum drifts rapidly (as in label permutations), maintaining low dynamic regret requires sufficient **tracking capacity**—an effective step-size large enough to follow the moving target. This trade-off between stability (protecting learned solutions) and plasticity (tracking new optima) directly informs our analysis of when and where utility gating should be applied.

### Sparse and Selective Adaptation

Our finding that only a **tiny fraction** (~0.3%) of parameters carry extreme utility connects to a broader literature on sparse learning. The **Lottery Ticket Hypothesis** [@Frankle2019] demonstrated that dense networks contain sparse subnetworks capable of matching full performance when trained in isolation. This suggests that effective learning may concentrate in a small subset of parameters—consistent with our observation that utility is highly concentrated.

In transfer learning, parameter-efficient fine-tuning methods like **LoRA** [@Hu2022] and **Adapters** [@Houlsby2019] achieve strong performance by modifying only a small fraction of parameters. These approaches implicitly recognize that adaptation pressure concentrates in specific network components. Our work provides a complementary perspective: rather than *designing* sparsity through architectural constraints, we show that *emergent* utility-based sparsity naturally identifies which parameters require protection versus exploration.

### Curvature-Aware Optimization

Beyond Lewandowski's curvature explanation, a related line of work connects optimization geometry to generalization. **Sharpness-Aware Minimization (SAM)** [@Foret2021] explicitly seeks flat minima by minimizing worst-case loss in a neighborhood, improving generalization under distribution shift. The connection between flat minima and generalization traces back to @Hochreiter1997, who argued that broad optima correspond to lower description complexity.

UPGD-W's utility-based gating can be understood through this lens: high-utility parameters may correspond to directions where the loss surface is curved (sensitive to perturbation), while low-utility parameters lie in flatter directions where noise injection causes minimal disruption. This interpretation complements Lewandowski's direct curvature measurement with a computationally efficient proxy, though the precise relationship between utility and local curvature merits further theoretical investigation.

## Background

### Problem Setting

We consider online continual supervised learning over a stream of examples $(x_{t}, y_{t})$. Let $f_\theta$ denote a neural network with parameters $\theta \in \mathbb{R}^N$ and loss function $\ell(f_\theta(x), y)$. The learning algorithm receives minibatches sequentially and must update $\theta$ to minimize cumulative loss.

Following @Lewandowski2024 and @Kumar2023, we focus on settings where the data distribution changes periodically. Every $U$ updates, the distribution shifts---either through input permutation (testing adaptation to feature changes) or label permutation (testing adaptation to output mapping changes). Unlike forgetting-focused benchmarks, we measure performance on the *current* task rather than retention of previous tasks. This design choice isolates **trainability** (the ability to learn new tasks) from **retention** (preserving old knowledge), which are distinct phenomena that may require different interventions [@Klein2024Survey; @Berariu2021].


We define plasticity following @Lyle2023 as the ability to achieve low error on new tasks. Specifically, if $J_{K}$ denotes the error at the end of task $K$, loss of plasticity occurs when $J_{K}$ increases with $K$---the network becomes progressively worse at learning, even though each task is equally difficult for a randomly initialized network.

### Shrink \& Perturb Baseline

The Shrink \& Perturb method [@Ash2020] serves as our primary baseline. It modifies standard SGD by shrinking weights and adding isotropic noise:
\[
\theta \leftarrow (1-\eta\lambda)\theta - \eta\,(g_{t} + \sigma \varepsilon_{t}), \quad \varepsilon_{t} \sim \mathcal{N}(0, I)
\]
where $\eta$ is the learning rate, $\lambda$ controls weight decay (shrinkage), and $\sigma$ controls noise magnitude.

@Ash2020 showed that this simple modification closes the warm-start generalization gap when $\lambda$ and $\sigma$ are properly tuned. The mechanism is attributed to balancing gradient contributions: shrinking increases the loss on previously seen data, making its gradients comparable in magnitude to new data. The noise prevents the optimizer from settling into narrow regions that generalize poorly.

However, S\&P has a fundamental limitation: it treats all parameters identically. Important parameters that encode useful features receive the same perturbation as redundant or poorly-utilized parameters. This motivates utility-based approaches that selectively apply perturbation.

## Method

### Overview

UPGD-W extends the S\&P framework by introducing **parameter-level utility scores** that **selectively gate noise injection into parameter updates**. 
The key insight is that not all parameters contribute equally to current performance. Parameters with high utility---those whose current values are important for minimizing loss---should be protected from large updates. Parameters with low utility can safely be perturbed to explore alternative solutions.

Unlike second-order importance measures based on Fisher information [@Kirkpatrick2017] or Hessian eigenvalues [@Lewandowski2024], UPGD-W utilizes a **first-order utility proxy** in its standard formulation to maintain computational efficiency. While the original UPGD framework proposed parameter-specific second-order utility estimates (improving performance in some settings), the first-order approximation is widely used to ensure $O(N)$ scalability and fair comparison with other first-order methods. This proxy relies only on gradient and parameter values, minimizing overhead while capturing the essential notion of parameter importance.

### Utility Estimation

UPGD-W estimates utility using the product of gradient and parameter value:
\[
\tilde{u}_{t,i} = - g_{t,i} \cdot \theta_{t,i}
\]
This quantity has an intuitive interpretation: when the gradient $g_{t,i}$ and parameter $\theta_{t,i}$ have opposite signs, their product is positive (after negation), indicating that the current parameter value is contributing to loss reduction. 
<!-- This intuitive interpretation is that high utility indicates the current parameter contributes to loss reduction. This is the right interpretation.  -->
This occurs when the parameter is "correctly positioned" relative to the current gradient direction.

To smooth noisy estimates, we maintain an exponential moving average (EMA) with decay factor $\beta \in (0, 1)$ and bias correction:
\[
\bar{u}_{t,i} = \beta \bar{u}_{t-1,i} + (1-\beta)\tilde{u}_{t,i}, \qquad
\hat{u}_{t,i} = \frac{\bar{u}_{t,i}}{1-\beta^t}
\]

The bias correction follows @Kingma2015 and ensures accurate estimates in early training when the EMA has not yet converged.

### Utility Normalization

Raw utility values span an unbounded range, making them difficult to use directly for gating. 
The original UPGD framework [@Elsayed2024] proposed both local (layer-wise) and global (network-wide) maximum scaling methods for utility normalization. In this work, we focus on the **global maximum scaling** approach combined with sigmoid transformation:
\[
u_{\max,t} = \max_{i} \hat{u}_{t,i}, \qquad
s_{t,i} = \sigma\!\left(\frac{\hat{u}_{t,i}}{\max(u_{\max,t}, \epsilon)}\right)
\]

The sigmoid function is used here to map the scaled utilities to $(0, 1)$, providing smooth, differentiable gating with a natural center at 0.5. While not strictly necessary—alternative scaling functions such as linear clipping or softmax could be used—sigmoid offers advantages: it is smoothly differentiable everywhere, naturally handles both positive and negative utilities, and provides a soft threshold that avoids hard cutoffs. However, as we discuss in Section 4.4.2, this approach can lead to concentration of utilities in a narrow range when raw values are not well-distributed, though this does not prevent effective gating in practice. The normalized utility $s_{t,i} \in (0, 1)$ represents the relative importance of parameter $i$. Parameters near the global maximum utility receive $s \approx 1$ (high protection), while parameters with lower or negative utility receive $s \approx 0.5$ or below (low protection).

### Gated Update Rule

The UPGD-W update combines gradient descent with utility-gated noise and weight decay. Define the gate $a_{t,i} = 1 - s_{t,i} \in (0, 1)$:
\[
\theta_{t+1,i} = (1-\eta\lambda)\theta_{t,i} - \eta\,a_{t,i}\,(g_{t,i} + \varepsilon_{t,i}), \quad \varepsilon_{t,i} \sim \mathcal{N}(0,1)
\]

The gate $a_{t,i}$ scales the effective update:
- **High-utility parameters** ($s \to 1$, $a \to 0$): Receive near-zero updates, preserving their current values
- **Low-utility parameters** ($s \to 0$, $a \to 1$): Receive full updates including noise, enabling exploration

This selective mechanism allows UPGD-W to maintain plasticity in low-utility regions while protecting learned representations encoded in high-utility parameters.

### Utility Distribution Tracking

To understand how utility values evolve during training and their role in gating effectiveness, we track the distribution of scaled utility values $s_{t,i}$ across all parameters. We partition the $[0, 1]$ range into nine bins with finer resolution around 0.5 (where gating has intermediate effect): $[0, 0.2)$, $[0.2, 0.4)$, $[0.4, 0.44)$, $[0.44, 0.48)$, $[0.48, 0.52)$, $[0.52, 0.56)$, $[0.56, 0.6)$, $[0.6, 0.8)$, and $[0.8, 1.0]$.

This histogram methodology serves two purposes:
1. **Characterizing utility dynamics:** We observe how the utility distribution shifts across task boundaries and training phases, revealing whether parameters naturally develop the bimodal structure (high vs. low utility) that UPGD-W assumes.
2. **Informing ablation design:** The observed distribution guides our clamping experiments—by constraining utilities to specific ranges, we can isolate the contribution of different parts of the distribution to overall performance.

### Utility Clamping Variants
<!-- This section might be better to move to the data interpretation section because this 0.52 is a clamping threshold based on the experiment results-->
To understand the role of utility dynamics, we evaluate variants that constrain utility values:

- **Clamped [0, 0.52]:** Maximum utility capped at 0.52
- **Clamped [0.44, 0.56]:** Symmetric clamping around 0.5
- **Clamped [0.48, 0.52]:** Tight symmetric clamping

These ablations test whether the natural utility tail (parameters with extreme utility values) is essential for effective gating, or whether a more uniform treatment suffices.

### Layer-Selective Variants
<!-- This section also would be better to move to the data interpretation section as the clamping section -->
Based on evidence that plasticity loss can concentrate in late layers [@Klein2024Survey; @Nikishin2022] and that memorization localizes differently across network depth [@Anagnostidis2023; @Maini2023], we investigate layer-selective gating:

- **UPGD-W (Full):** Utility gating applied to all layers
- **UPGD-W (Output Only):** Gating applied only to the final classification layer; hidden layers use fixed scaling (0.5) like standard SGD
- **UPGD-W (Hidden Only):** Gating applied only to hidden layers; output layer uses fixed scaling
- **UPGD-W (Hidden+Output):** Explicit control over each layer type

The rationale is that different layers may have different plasticity requirements. Output layers must adapt to changing label mappings in label-permuted tasks, while hidden layers may benefit from stable feature representations.

### Theoretical Motivation

We provide theoretical grounding for two key empirical patterns:

**(S1) Extreme-utility parameters are sparse and concentrate in the output layer.** Across all methods, ~99.7% of parameters have near-neutral utility (0.48–0.52), while a sparse subset (~0.3%) carries extreme utility. Critically, per-layer analysis reveals that this sparse subset is more prevalent in the output layer, reinforcing why output-selective gating is effective.

**(S2) Output-only gating dominates on target-shift streams.** Under target shift, the head optimum moves significantly while the representation optimum is approximately stable. Gating only the output layer allows hidden layers to remain plastic for representation learning.

Together, these findings suggest that utility-based protection should be *layer-selective*: the output layer experiences the highest adaptation pressure under target shift, develops the most extreme-utility parameters, and thus benefits most from protective gating.

#### Online Learning Preliminaries

Before presenting our propositions, we briefly review concepts from online convex optimization [@Zinkevich2003] that ground our theoretical analysis.

In online learning, a learner makes sequential decisions $w_{1}, w_{2}, \ldots, w_{T}$ and incurs losses $\ell_{1}(w_{1}), \ell_{2}(w_{2}), \ldots$. Performance is measured by **regret**---the gap between cumulative loss and a comparator.

**Static regret** compares to the single best fixed decision in hindsight:
\[
\text{Static Regret} = \sum_{t=1}^{T} \ell_{t}(w_{t}) - \min_{w^*} \sum_{t=1}^{T} \ell_{t}(w^*)
\]

**Dynamic regret** [@Besbes2015] compares to a sequence of time-varying optimal solutions, appropriate for non-stationary environments where the optimum itself changes over time:
\[
\text{Dynamic Regret} = \sum_{t=1}^{T} \ell_{t}(w_{t}) - \sum_{t=1}^{T} \ell_{t}(w^*_{t})
\]
where $w^*_{t}$ is the optimal parameter at each time step $t$.

A key result from this literature is that dynamic regret scales with the **path length** of the comparator sequence: $\sum_{t} \|w^*_{t+1} - w^*_{t}\|$. When the optimum drifts rapidly (as in label permutations where the optimal output head $W^*$ changes abruptly at each task boundary), 
maintaining low dynamic regret requires sufficient **tracking capacity**---an effective step-size large enough to follow the moving target.

This framework directly applies to our setting: under target shift, the optimal output weights form a non-stationary comparator sequence. An optimizer's ability to track this sequence---rather than converge to a fixed point---determines its continual learning performance.

#### Notation for Theoretical Analysis

We use the following notation throughout (formal definitions in Appendix):

| Symbol | Definition |
|--------|------------|
| $\theta = (V, W)$ | Parameters: hidden layers $V \in \mathbb{R}^{d_V}$, output layer $W \in \mathbb{R}^{d_W}$ |
| $g_{t,i}$ | Gradient $\partial \ell_t / \partial \theta_i$ |
| $\tilde{u}_{t,i} = -g_{t,i} \cdot \theta_{t,i}$ | Raw utility (first-order importance) |
| $\hat{u}_{t,i}$ | Bias-corrected utility (EMA with decay $\beta$) |
| $s_{t,i} = \sigma(\hat{u}_{t,i}/u_{\max,t})$ | Normalized utility $\in (0,1)$ after sigmoid |
| $a_{t,i} = 1 - s_{t,i}$ | Gate (scaling factor for updates) |
| $\mathcal{I}^+_t, \mathcal{I}^-_t, \mathcal{B}_t$ | High-utility (protected), low-utility (exploratory), bulk sets |

The gate $a_{t,i}$ modulates the effective learning rate: high-utility parameters ($s \to 1$, $a \to 0$) receive reduced updates (protection), while low-utility parameters ($s \to 0$, $a \to 1$) receive amplified updates (exploration).

#### Proposition A: Effective Tail-Based Gating

Global-max normalization combined with sigmoid transformation creates a characteristic gating structure where most parameters receive near-neutral treatment while small fractions at both extremes receive differential treatment. Because $u_{\max,t} = \max_{i} \hat{u}_{t,i}$, the ratio $r_{t,i} = \hat{u}_{t,i}/u_{\max,t} \in [0,1]$ concentrates near zero for the vast majority of parameters. After the sigmoid maps to bounded $s_{t,i} \in (0,1)$:
- Most parameters get $s \approx \sigma(0) = 0.5 \Rightarrow a \approx 0.5$ (neutral gating)
- High-utility tail gets $s > 0.5 \Rightarrow a < 0.5$ (reduced updates, protection)
- Low-utility tail gets $s < 0.5 \Rightarrow a > 0.5$ (amplified updates, exploration)

\begin{proposition}[Effective tail-based gating]
Define the extreme sets: $\mathcal{I}^{+}_{t}(\tau) = \{i : s_{t,i} \geq 0.5 + \tau\}$ (high-utility, protected) and $\mathcal{I}^{-}_{t}(\tau) = \{i : s_{t,i} \leq 0.5 - \tau\}$ (low-utility, exploratory). Under the empirically observed utility concentration, $|\mathcal{I}^{+}_{t}| + |\mathcal{I}^{-}_{t}| \ll N$ while the bulk satisfies $a_{t,i} \approx 0.5$. Therefore, UPGD-W behaves as:
\[
\text{baseline noisy SGD} + \text{selective modulation on extreme-utility tails}
\]
The algorithm's deviation from constant-step noisy SGD is controlled by the combined tail mass and the spread of the utility distribution around the neutral point.
\end{proposition}

**Dual role of tails.** The high-utility tail ($s > 0.5$) receives protection through reduced update magnitude, preserving parameters critical for current performance. The low-utility tail ($s < 0.5$) receives amplified updates including noise, enabling exploration of alternative parameter configurations. Both mechanisms contribute to continual learning: protection maintains stability while amplified exploration sustains plasticity.

**Implication for clamping.** Clamping compresses utilities toward 0.5, eliminating both tails. This removes the algorithm's ability to differentiate parameters---neither protecting critical ones nor exploring underutilized ones. If performance depends on this selective treatment (as our experiments suggest), then constraining the utility distribution inevitably degrades accuracy.

**Empirical support.** Our utility histogram analysis (Section 4.3) confirms that 99.7–100\% of parameters have utilities in $[0.48, 0.52]$, with the $<0.3\%$ in both tails driving effective parameter selection. Notably, clamping to $[0, 0.52]$ (preserving low-utility tail) outperforms symmetric clamping $[0.44, 0.56]$, suggesting the low-utility tail's exploratory role is as important as the high-utility tail's protective role. Per-layer analysis shows that extreme utilities concentrate in the output layer, consistent with head-centric adaptation pressure under target shift.

#### Proposition B: Output-Only Advantage Under Target Shift

Model the network as a representation plus head: $f(x; V, W) = W \phi(x; V)$, where $V$ parameterizes hidden layers and $W$ the output layer. Consider cross-entropy training on a stream with piecewise-stationary tasks.

**target-shift regime (label permutations).** Under label permutation, the conditional $P(y \mid x)$ changes abruptly while the input distribution $P(x)$ remains fixed---a form of target non-stationarity in the terminology of @Klein2024Survey. Formally:
- The feature distribution over $\phi(x; V)$ is roughly stable across tasks (or changes slowly). This assumption is supported by prior work showing that shallow layers converge to general, transferable features [@Yosinski2014; @Wei2024], though we do not empirically verify representation stability in this work.
- The optimal head $W^*_{k}$ changes abruptly across tasks $k$ (often exactly by a permutation of class labels).

\begin{proposition}[Output-only advantage]
In a target-shift stream where the head optimum $W^*_{k}$ moves significantly across tasks while the representation optimum $V^*$ is approximately stable, applying utility-gated scaling only to $W$ yields a smaller dynamic-regret upper bound than gating both $W$ and $V$.
\end{proposition}

**Proof sketch.** (Full proof in Appendix.)
The head update is preconditioned noisy SGD: $W_{t+1} = W_{t} - \eta A_{t}(\nabla \ell_{t}(W_{t}) + \sigma \varepsilon_{t})$, where $A_{t} = \text{diag}(a_{t,i})$ with $a_{t,i} \in (0,1)$. By standard dynamic regret analysis [@Zinkevich2003; @Hazan2016], tracking a drifting comparator $W^*_t$ requires sufficient effective step-size. Under target shift, the path length $P_T = \sum_t \|W^*_{t+1} - W^*_t\|$ is large (head must track label permutations), and the resulting upper bound deteriorates as the effective step-size shrinks (e.g., scaling like $O(D\,P_T/(\eta a_{\min}))$ for a diameter bound $D$).

Full gating reduces $a_{\min}$ on high-utility head parameters *and* degrades feature quality through constrained hidden layer adaptation. Let $\tilde{G}_W = G_W + L_\phi\|\phi^{\text{full}} - \phi^*\|$ denote the effective gradient bound under degraded features. The feature degradation penalty is:
$$\Delta_{\text{feature}}(T) = \frac{\eta T}{2}\left(\tilde{G}_W^2 - G_W^2\right) > 0$$
when full gating prevents optimal feature adaptation.

**Where output-only wins.** Full gating damps hidden layer updates on high-utility coordinates, preventing features from maintaining class separability. This "feature drift penalty" makes the head optimization problem harder (larger effective gradient bound, slower tracking). Output-only gating avoids this by maintaining uniform $a^V_i = 0.5$ on hidden layers while still applying selective gating to the head, yielding $\overline{R}^{\text{out}}_T \leq \overline{R}^{\text{full}}_T - \Delta_{\text{feature}}(T)$ when non-stationarity is head-dominated.

#### Synthesis: Why Output-Only Gating Succeeds Under Target Shift

Combining these propositions clarifies the mechanism behind our main empirical result on label-permuted benchmarks. **Proposition A** (the local mechanism) explains *how* the output layer $W$ can be effectively gated: the tail-based structure ensures that high-utility parameters receive protection while low-utility parameters receive amplified exploration. Both tails contribute---protection maintains task-critical mappings while exploration sustains adaptability. **Proposition B** (the global strategy) explains *why* hidden layers should remain ungated under target shift: allowing the representation $V$ to adapt freely avoids a feature drift penalty when the optimal representation is approximately stable.

Together, UPGD-W (Output Only) achieves an optimal balance for target-shift scenarios: it maximizes plasticity in the representation (via non-gating) and provides selective modulation in the head (via tail-based gating). For input-shift scenarios where representation relearning is primary, this advantage diminishes and full-network gating may be equally appropriate.

#### Scope of Theoretical Results

**Propositions A and B are specific to target-shift (label-permutation) scenarios.** The theoretical advantage of output-only gating relies on the assumption that the optimal representation is approximately stable while the optimal head changes (Proposition B, assumption B2). Under input shift---where representation relearning is the primary challenge---this assumption does not hold, and consequently:

- The "feature drift penalty" argument (Proposition B) loses force when features must adapt
- Layer-selective advantages disappear; full-network gating becomes competitive
- Empirically, Input-Permuted MNIST shows minimal difference between Output Only and Full variants

This scope limitation is not a weakness but a design principle: **match the intervention to the non-stationarity type**. For target shift, focus gating on the output layer; for input shift, consider full-network approaches or methods that explicitly support representation relearning. The broader implication is that no single gating strategy dominates across all non-stationarity regimes---effective continual learning may require detecting or anticipating the shift type and adapting the intervention accordingly.

## Experiments

### Experimental Setup

**Datasets.** We evaluate on four continual learning benchmarks with increasing complexity following the experimental setup of @Elsayed2024:

1. **Input-Permuted MNIST** [@LeCun2010; @Goodfellow2013]: A plasticity-focused benchmark where input pixels are randomly permuted every 5,000 steps. This tests adaptation to changing input statistics without altering the classification task. We train for 1M steps (200 permutation changes).

2. **Label-Permuted EMNIST** [@Javed2019]: A 47-class character recognition task where label assignments are randomly permuted between tasks. This creates abrupt distribution shifts that challenge continual learning methods. We use 400 tasks with 2,500 samples per task (1M total steps).

3. **Label-Permuted CIFAR-10** [@Krizhevsky2009; @Kumar2023]: A visual classification task with 10 classes. Label permutations test the model's ability to reorganize output mappings while maintaining input representations. Same task structure as EMNIST.

4. **Label-Permuted Mini-ImageNet** [@Dohare2023]: The most challenging benchmark, using 2048-dimensional ResNet-50 features and 100 classes. This tests scalability to higher-dimensional inputs and larger output spaces.

<!-- I need to check @Dohare2023 and @Kumar2023 - what architecture do they use because I have the code -->
**Architecture.** Following @Elsayed2024, we use a fully-connected ReLU network: input $\to$ 300 $\to$ 150 $\to$ output. This simple architecture isolates optimizer effects from architectural confounds. All experiments use online learning (batch size 1).
<!-- I need to begin again my writing papers from this point -->
These benchmarks are intentionally chosen to probe different non-stationarity types discussed in the plasticity-loss literature (input shifts vs. target shifts) under a controlled supervised setting, rather than to model the full complexity of RL training dynamics (exploration, replay, and bootstrapped targets) [@Klein2024Survey].

**Baselines.** We compare against **Shrink & Perturb (S&P)** [@Ash2020], which applies uniform noise injection with weight decay.

**UPGD-W Variants.** We evaluate multiple configurations of UPGD-W, including:
- Layer-selective variants (Output Only, Hidden Only, Full)
- Utility-clamped configurations

**Metrics.** We report final accuracy (averaged over the last 10% of training), plasticity (accuracy improvement within each task), and dead unit fraction (neurons with zero activation).

### Main Results

#### Label-Permuted Mini-ImageNet

Mini-ImageNet represents our most challenging benchmark, combining high-dimensional inputs (2048D) with a large output space (100 classes) and abrupt label changes.

 ![Training dynamics on Mini-ImageNet comparing accuracy, loss, and plasticity across methods.](figures/mini_imagenet/accuracy_comparison.png){#fig:mini_imagenet_accuracy width=90%}

**Results.** UPGD-W (Output Only) achieves 0.6918 accuracy compared to S\&P's 0.2810 (+146\%). This is our strongest result, demonstrating the effectiveness of output-selective gating on complex label-permuted tasks.

#### Label-Permuted EMNIST

EMNIST (47 classes) provides an intermediate challenge between MNIST and Mini-ImageNet.

 ![Training dynamics on Label-Permuted EMNIST comparing accuracy, loss, and plasticity across methods.](figures/emnist/accuracy_comparison.png){#fig:emnist_accuracy width=90%}

**Results.** UPGD-W (Output Only) achieves 0.7700 accuracy compared to S\&P's 0.3323 (+132\%). The pattern mirrors Mini-ImageNet: output-only gating outperforms full gating (0.7271).

#### Label-Permuted CIFAR-10

CIFAR-10 tests continual learning on natural images with a smaller output space (10 classes).

 ![Training dynamics on Label-Permuted CIFAR-10 comparing accuracy, loss, and plasticity across methods.](figures/cifar10/accuracy_comparison.png){#fig:cifar10_accuracy width=90%}

**Results.** UPGD-W (Hidden+Output) achieves the best accuracy at 0.6187 (+66\% over S\&P's 0.3716), with Output Only close behind at 0.5971 (+61\%). The smaller improvement compared to Mini-ImageNet/EMNIST may reflect the smaller output space (10 classes).

#### Input-Permuted MNIST

Input-Permuted MNIST tests adaptation to changing input statistics. Unlike the label-permuted benchmarks above, the classification task remains fixed---only the pixel ordering changes. This represents a fundamentally different type of non-stationarity (input shift vs. target shift).

 ![Training dynamics on Input-Permuted MNIST comparing accuracy, loss, and plasticity across methods.](figures/input_mnist/accuracy_comparison.png){#fig:input_mnist_accuracy width=90%}

**Results.** UPGD-W (Output Only) achieves 0.7809 accuracy compared to S\&P's 0.7442 (+5\%), while UPGD-W (Full) achieves 0.7707. The marginal differences between layer-selective variants reflect the fixed output mapping: since labels don't change, the theoretical advantage of output-only gating (Proposition B) does not apply. Under input shift, where representation relearning is the primary challenge, full-network gating may be equally or more appropriate. This aligns with the distinction between input-shift and target-shift regimes emphasized by @Klein2024Survey.

### Cross-Dataset Analysis

This section consolidates our analysis across all four benchmarks, organizing findings by topic to reveal consistent patterns and dataset-specific variations.

#### Summary of Results

 Table~\ref{tbl:best_method_summary} summarizes the best-performing method and improvement over S\&P across all benchmarks.

 | Dataset | Best Method | Best Acc | S\&P Acc | Improvement |
 |---------|-------------|----------|---------|-------------|
 | Input-MNIST | UPGD-W (Output Only) | 0.7809 | 0.7442 | +5\% |
 | Mini-ImageNet | UPGD-W (Output Only) | 0.6918 | 0.2810 | +146\% |
 | EMNIST | UPGD-W (Output Only) | 0.7700 | 0.3323 | +132\% |
 | CIFAR-10 | UPGD-W (Hidden+Output) | 0.6187 | 0.3716 | +66\% |
 : Best-performing method summary across benchmarks. {#tbl:best_method_summary}

**Consistent Pattern.** UPGD-W (Output Only) achieves the best or near-best performance on three of four benchmarks, with Hidden+Output slightly outperforming on CIFAR-10. The improvement magnitude correlates with task complexity: larger gains on difficult label-permuted tasks with more classes.

#### Layer-Selective Gating

We systematically evaluate which layers benefit from utility gating.

![Layer-selective gating accuracy comparison across datasets.](figures/cross_dataset/layer_selective_comparison.png){#fig:layer_selective_comparison width=90%}

 | Gating Strategy | Mini-ImageNet | EMNIST | CIFAR-10 | Input-MNIST |
 |-----------------|---------------|--------|----------|-------------|
 | Full (all layers) | 0.6150 | 0.7271 | 0.5955 | 0.7707 |
 | Output Only | **0.6918** | **0.7700** | 0.5971 | **0.7809** |
 | Hidden Only | 0.0464 | 0.3757 | 0.4339 | -- |
 | Hidden+Output | 0.6198 | 0.7315 | **0.6187** | -- |
 : Layer-selective gating accuracy comparison. {#tbl:layer_selective}

 | Dataset | Input Dim | Output Dim | Output Params | Total Params | Output \% |
 |---------|-----------|------------|---------------|--------------|-----------|
 | Input-MNIST | 784 | 10 | 1,510 | 282,310 | 0.53\% |
 | EMNIST | 784 | 47 | 7,097 | 287,897 | 2.47\% |
 | CIFAR-10 | 3,072 | 10 | 1,510 | 969,310 | 0.16\% |
 | Mini-ImageNet | 2,048 | 100 | 15,100 | 676,000 | 2.23\% |
 : Percentage of parameters in output layer (gated in Output Only). {#tbl:percent_gated}

**Key Insight.** The striking finding is that output-only gating—which applies utility-based protection to only **0.16--2.5\%** of network parameters (Table~\ref{tbl:percent_gated})—is sufficient to achieve up to 146\% improvement over S\&P. This supports the hypothesis that for label-permuted tasks, the output layer encodes task-specific mappings requiring selective protection, while hidden layers benefit from unrestricted plasticity to learn transferable features.

**Additional Insights:**

1. **Output layer gating is critical.** Removing it (Hidden Only) causes catastrophic degradation: -93\% on Mini-ImageNet, -51\% on EMNIST. This supports the hypothesis that output layers encode task-specific mappings requiring careful protection.

2. **Hidden layer gating provides marginal benefit.** Comparing Output Only vs Full, adding hidden layer gating slightly *decreases* performance on challenging tasks. Hidden layers may benefit from unrestricted plasticity to learn transferable features.

3. **The pattern is consistent across datasets.** Despite different input modalities and output spaces, output-only gating consistently outperforms alternatives.

#### Utility Clamping

We test whether constraining utility dynamics affects performance.

![Utility clamping effect on accuracy across datasets.](figures/cross_dataset/clamping_degradation.png){#fig:clamping_degradation width=90%}

 | Clamping Range | Mini-ImageNet | EMNIST | CIFAR-10 |
 |----------------|---------------|--------|----------|
 | Unclamped | **0.6150** | **0.7271** | **0.5955** |
 | [0, 0.52] | 0.4555 (-26\%) | 0.7046 (-3\%) | 0.5540 (-7\%) |
 | [0.44, 0.56] | 0.4010 (-35\%) | 0.5426 (-25\%) | 0.5605 (-6\%) |
 | [0.48, 0.52] | 0.3029 (-51\%) | 0.3711 (-49\%) | 0.5248 (-12\%) |
 : Effect of utility clamping on accuracy. {#tbl:utility_clamping}

**Key Insights:**

1. **Natural utility dynamics are essential.** Performance depends not just on constraint tightness but on which utilities are accessible. Clamping to [0, 0.52] (preserving the low-utility tail) outperforms [0.44, 0.56] (symmetric but narrower), suggesting the algorithm benefits from access to both extremes of the utility distribution.

2. **The utility tail matters.** Although 99.7\%+ of parameters have utilities in [0.48, 0.52], the small fraction with extreme values drives effective parameter selection.

3. **Constraints interact with task difficulty.** Complex tasks (Mini-ImageNet, EMNIST) suffer more from clamping than simpler tasks (CIFAR-10).

#### Utility Decay Factor ($\beta$)

The decay factor controls temporal smoothing of utility estimates.

 | Dataset | Optimal $\beta$ | Effective Memory |
 |---------|-----------------|------------------|
 | Input-MNIST | 0.9999 | ~10,000 steps |
 | EMNIST | 0.9 | ~10 steps |
 | CIFAR-10 | 0.999 | ~1,000 steps |
 | Mini-ImageNet | 0.9 | ~10 steps |
 : Optimal utility decay factor by dataset. {#tbl:utility_decay}

**Interpretation:** Input-permuted tasks benefit from long memory ($\beta$=0.9999) because useful representations persist across permutations. Label-permuted tasks require rapid utility adaptation ($\beta$=0.9) because parameter importance changes dramatically when output mappings shift.

#### Dead Unit Patterns

Dead units (neurons with zero activation) indicate lost network capacity. We compare dead unit fractions across methods and datasets.

![Dead unit fraction comparison across datasets and methods.](figures/cross_dataset/dead_units_comparison.png){#fig:dead_units_comparison width=90%}

 | Method | Mini-ImageNet | EMNIST | CIFAR-10 | Input-MNIST |
 |--------|---------------|--------|----------|-------------|
 | UPGD-W (Output Only) | 0.8633 | 0.8141 | 0.7876 | 0.5231 |
 | UPGD-W (Full) | 0.7668 | 0.7318 | 0.7799 | 0.5891 |
 | UPGD-W (Hidden+Output) | 0.7636 | 0.7242 | 0.7786 | 0.5244 |
 | S\&P | 0.9464 | 0.9431 | 0.7876 | 0.7942 |
 : Dead unit fraction comparison (Final Avg). Lower is better. {#tbl:dead_units}

**Key Observations:**

1. **S\&P produces the most dead units on most benchmarks.** On Mini-ImageNet (0.9464), EMNIST (0.9431), and Input-MNIST (0.7942), S\&P has significantly higher dead unit fractions than UPGD-W variants. This suggests uniform noise injection is less effective at preserving network capacity.

2. **UPGD-W (Hidden+Output) and Full achieve lowest dead units on label-permuted tasks.** These variants show 0.72--0.78 dead unit fractions, approximately 20% lower than S\&P on Mini-ImageNet and EMNIST.

3. **Output Only has higher dead units but better accuracy.** Despite having more dead units than Full/Hidden+Output variants, Output Only achieves better accuracy. This apparent paradox suggests that beyond a certain threshold, additional live units provide diminishing returns---the quality of parameter selection matters more than raw capacity.

4. **Input-MNIST shows reversed pattern.** S\&P has high dead units (0.7942) while UPGD-W variants maintain lower fractions (0.52--0.59). This may reflect different optimization dynamics under input-shift versus target-shift non-stationarity.

#### Plasticity Patterns

Plasticity measures the ability to improve within each task. We compare plasticity metrics across methods and datasets.

![Plasticity comparison across datasets and methods.](figures/cross_dataset/plasticity_comparison.png){#fig:plasticity_comparison width=90%}

 | Method | Mini-ImageNet | EMNIST | CIFAR-10 | Input-MNIST |
 |--------|---------------|--------|----------|-------------|
 | UPGD-W (Output Only) | 0.5069 | 0.4327 | 0.3876 | 0.4961 |
 | UPGD-W (Full) | 0.5558 | 0.5065 | 0.4125 | 0.5177 |
 | UPGD-W (Hidden+Output) | 0.5524 | 0.4998 | 0.4312 | 0.4963 |
 | S\&P | 0.4119 | 0.4849 | 0.3799 | 0.5773 |
 : Plasticity comparison (Final Avg). Higher is better. {#tbl:plasticity}

**Key Observations:**

1. **Higher plasticity does not guarantee higher accuracy.** S\&P shows relatively high plasticity on Input-MNIST (0.5773) and EMNIST (0.4849), yet achieves lower accuracy than UPGD-W variants. This suggests plasticity alone is insufficient---effective parameter selection determines whether plasticity translates to learning.

2. **UPGD-W (Full) shows highest plasticity on label-permuted tasks.** On Mini-ImageNet (0.5558) and EMNIST (0.5065), Full gating achieves the highest plasticity scores. However, Output Only achieves better accuracy despite lower plasticity, indicating that selective protection in the output layer matters more than overall plasticity.

3. **The plasticity-accuracy gap varies by task type.** On label-permuted tasks (Mini-ImageNet, EMNIST), the gap between plasticity and accuracy rankings is larger. On Input-MNIST, methods with higher plasticity (S\&P) do achieve reasonable accuracy, consistent with the different non-stationarity type.

#### Why Does Output-Only Gating Outperform Full Gating?

The consistent superiority of output-only gating across datasets warrants explanation. **Proposition B** (Theoretical Motivation) provides formal grounding: in target-shift streams, the head optimum $W^*_{k}$ moves significantly while the representation optimum is approximately stable. Full gating incurs a "feature drift penalty" by damping representation updates, making the head problem harder. We identify three contributing mechanisms:

1. **Output layer encodes task-specific information.** In label-permuted tasks, the output layer must map features to changing label assignments. Protecting high-utility output weights preserves correct mappings, while allowing free hidden layer adaptation enables feature refinement. This directly validates the target-shift assumption in Proposition B.

2. **Hidden layers benefit from plasticity.** Hidden layers learn general feature representations that transfer across tasks. Constraining their updates with utility gating may prevent beneficial adaptation to new input distributions. This aligns with evidence that shallow layers converge to general, transferable features while deep layers specialize to task-specific content [@Wei2024; @Anagnostidis2023]. In the language of Proposition B, gating $V$ reduces tracking capacity for representation coordinates that must maintain separability.

3. **Utility estimation is more reliable at the output.** The output layer receives direct gradient signal from the loss, making utility estimates more accurate. Hidden layer utilities are attenuated through backpropagation, potentially introducing noise that harms gating decisions. Our per-layer utility statistics confirm this: the output layer shows 2--7$\times$ larger raw utility maxima and 2--98$\times$ larger high-utility shoulder mass than hidden layers across datasets.

**Connection to Curvature.** @Lewandowski2024 showed that loss of curvature directions explains plasticity loss. Output-only gating may preserve curvature in hidden layers while selectively protecting output layer structure. This hypothesis merits future investigation with explicit curvature measurements.

#### Utility Distribution Dynamics

Analysis of utility histograms reveals extreme concentration: 99.7--100\% of parameters have normalized utilities in $[0.48, 0.52]$. Figure~\ref{fig:utility_distribution_comparison} shows the utility distribution (log scale) at the end of training across all four datasets.
<!-- no need to show s&p; -->
![Utility distribution comparison across datasets (log scale). Each panel shows the percentage of parameters in each utility bin for different methods.](figures/cross_dataset/utility_distribution_comparison.png){#fig:utility_distribution_comparison width=90%}

![Detailed utility distributions (log scale) for each dataset individually.](figures/cross_dataset/utility_histogram_combined.png){#fig:utility_hist_combined width=90%}

**Key observations:**

1. **UPGD-W (Output Only)** (green triangles) consistently shows larger tails in both low-utility ($< 0.48$) and high-utility ($> 0.52$) bins compared to other methods, indicating that it more effectively distinguishes important from unimportant parameters.

2. **Clamped variants** exhibit truncated distributions as expected---the Clamped [0.48, 0.52] variant (brown hexagons) shows virtually no parameters outside the central bin, eliminating the utility-based selection mechanism.

The correlation between larger utility tails and better performance (particularly for Output Only) suggests that the ability to identify and differentially treat extreme-utility parameters is crucial for continual learning success.

**Proposition A** (Theoretical Motivation) formalizes this observation. Under global-max normalization with sigmoid transformation, the extreme sets $\mathcal{I}^{+}_{t}$ (high-utility) and $\mathcal{I}^{-}_{t}$ (low-utility) are tiny ($|\mathcal{I}^{+}_{t}| + |\mathcal{I}^{-}_{t}| \ll N$), while the vast majority receive neutral gating ($a \approx 0.5$). This structure has three implications:

1. **Gating operates at the margins.** Effective parameter selection relies on the $<0.3\%$ of parameters outside the central bin. These "outlier" utilities determine which parameters receive protection or amplified exploration. In Proposition A's terminology, only $\mathcal{I}^{+}_{t} \cup \mathcal{I}^{-}_{t}$ receives meaningful modulation.

2. **Global max normalization creates extreme contrast.** The sigmoid transformation after global max scaling compresses most utilities toward 0.5. Parameters near the global maximum receive strong protection ($s \to 1$), while low-utility parameters receive amplified exploration ($s < 0.5$). This is the "tail-based gating" mechanism characterized in Proposition A.

3. **Clamping disrupts the selection mechanism.** Constraining utilities to narrow ranges eliminates the outlier utilities that drive effective differentiation, explaining the severe performance degradation. Proposition A predicts this: clamping collapses both tails toward neutral, reducing UPGD-W to constant-step noisy SGD.

This confirms that UPGD-W's effectiveness stems from its ability to identify and differentially treat a small number of extreme-utility parameters (both high-utility for protection and low-utility for exploration), rather than from nuanced modulation across all parameters.

#### Per-Layer Utility Concentration

A critical question remains: *where* do these extreme-utility parameters concentrate? Per-layer utility analysis reveals a striking asymmetry between hidden and output layers that directly connects our two main findings (S1 and S2).

Figure~\ref{fig:per_layer_utility_comparison} shows the per-layer utility distributions across all four datasets, comparing hidden layers versus the output layer.

![Per-layer utility concentration comparison across datasets. Each panel shows the percentage of parameters in the high-utility tail (>0.52) for hidden vs. output layers.](figures/cross_dataset/per_layer_utility_comparison.png){#fig:per_layer_utility_comparison width=90%}

**Key Observations.**

1. **Output layer has more extreme utilities.** For UPGD-W (Full), the output layer shows substantially higher percentages of parameters in the high-utility tail (>0.52) compared to hidden layers. This asymmetry explains why output-only gating suffices: the parameters that matter most naturally concentrate in the output layer, where adaptation pressure is highest.

2. **Output-Only variant spreads utilities across layers.** When gating is applied only to the output layer, hidden layers also develop more extreme utilities. This likely reflects the increased freedom of ungated hidden layers to adapt, enabling more differentiated feature learning.

3. **The >0.52 tail is disproportionately located in the output layer.** This directly validates **(S1)**: extreme-utility parameters concentrate in the output layer—precisely where **(S2)** predicts adaptation pressure is highest under target shift.

**Synthesis.** This per-layer analysis provides the missing link between our two propositions. Under target shift, the output layer must rapidly track changing label mappings. Parameters that successfully track the shifting optimum acquire high utility, creating the extreme-utility tail. Because this tail naturally concentrates in the output layer, output-only gating—which protects precisely these parameters—is sufficient. Gating hidden layers provides no additional benefit because their utility distributions remain concentrated near neutral (0.5), indicating they lack the extreme-utility parameters that require protection.

#### Connection to Prior Work

Our findings align with and extend several lines of prior work:

**Curvature and plasticity [@Lewandowski2024].** The curvature-based explanation of plasticity loss suggests that networks lose optimization directions over time. UPGD-W's selective perturbation may preserve curvature by allowing continued exploration in low-utility regions while protecting the structure encoded by high-utility parameters.

**Regenerative regularization [@Kumar2023].** L2 regularization toward initialization preserves plasticity by keeping parameters near their starting distribution. UPGD-W achieves a similar effect through selective noise injection, but without explicit regularization that constrains parameter movement.

**Continual Backprop [@Dohare2023].** Neuron-level utility tracking and reinitialization inspired UPGD-W's parameter-level approach. The continuous gating mechanism avoids discrete reset decisions and provides finer-grained control.

**Dormant neurons [@Sokar2023].** Our dead unit analysis confirms that S\&P leads to higher dormant neuron fractions than UPGD-W. Selective perturbation appears to preserve network capacity better than uniform noise.


## Discussion

### Implications for Continual Learning

Our findings suggest several design principles for continual learning optimizers:

1. **Layer-selective treatment is essential.** Uniform treatment of all parameters, as in S\&P, fails to leverage the different roles of network components. Future methods should consider layer-specific or even parameter-specific strategies.

2. **First-order importance is sufficient.** UPGD-W achieves strong performance using only gradient-weight products, without Hessian computation. This suggests that computationally expensive second-order methods may not be necessary for effective continual learning.

3. **Natural dynamics should be preserved.** The severe degradation from utility clamping indicates that artificial constraints on learned representations are harmful. Algorithms should allow natural parameter dynamics to emerge from optimization.

4. **Output layer adaptation is critical.** For tasks involving output mapping changes, protecting and selectively adapting the output layer is more important than hidden layer management.

### Connection to Memorization

Recent work on memorization in deep learning [@Wei2024] reveals a striking parallel to our findings. The survey establishes that deep layers specialize to task-correlated features (effectively "memorizing" task-specific mappings), while shallow layers learn general, transferable patterns. Crucially, forgetting in continual learning preferentially affects these specialized deep-layer features---exactly where task-specific "memorization" resides [@Wei2024; @Anagnostidis2023].

This perspective reframes our results: output-only gating succeeds precisely because it protects the "memorization layer" (the output head encoding task-specific label mappings) while allowing hidden layers to maintain the general feature representations that support transfer. In continual learning under target shift, preserving this task-specific memorization is essential---catastrophic forgetting occurs when the output mapping is overwritten. UPGD-W's utility-based selection may therefore identify and protect exactly those parameters encoding content that would otherwise be catastrophically forgotten.

This connection suggests a broader principle: **effective continual learning requires balancing pattern learning (in hidden layers) with selective memorization protection (in output layers)**. Future work could explore whether the utility tail set $\mathcal{I}_{t}(\tau)$ corresponds to "memorization-critical" parameters identified through privacy or forgetting metrics.

### Interpreting Our Results Through Types of Non-Stationarity

Klein et al. argue that “plasticity loss” is not monolithic: the relevant bottleneck depends on whether the learning problem induces primarily **input non-stationarity** (changing representations) or **target non-stationarity** (changing label/target mappings), and whether the primary failure is **trainability** or **generalizability** [@Klein2024Survey]. This framing helps reconcile several patterns in our experiments.

First, our largest gains occur on **label-permuted** benchmarks (EMNIST, Mini-ImageNet), which are prototypical *target-shift* problems. In this regime, the survey suggests that many failures can be addressed by mechanisms that preserve or quickly restore head adaptability, consistent with empirical evidence that plasticity loss can concentrate in late layers in several settings [@Klein2024Survey; @Nikishin2022]. Our finding that **output-only gating dominates** is consistent with this: protecting tail (high or low)-utility output parameters while letting the representation adapt freely appears sufficient—and sometimes better than—gating the whole network.

Second, input-permuted MNIST is closer to a pure **input-shift** benchmark, where success requires relearning representations. Correspondingly, our improvements over S\&P are smaller, and the optimal \(\beta\) is much larger (longer utility memory), consistent with the idea that under input shifts, useful features (and their utilities) may persist longer even as the input mapping changes [@Klein2024Survey]. Together, these results suggest that layer-selective gating is not merely a computational trick but a way to match the intervention to the dominant form of non-stationarity.

### Relationship to Existing Methods

UPGD-W occupies a middle ground between several existing approaches:

**vs. EWC [@Kirkpatrick2017]:** Both methods identify important parameters, but EWC uses Fisher information (quadratic in gradient samples) while UPGD-W uses first-order products. EWC requires storing task-specific importance matrices; UPGD-W maintains a single running average.

**vs. S\&P [@Ash2020]:** S\&P applies uniform noise; UPGD-W **selectively gates noise injection** based on utility. This selective approach achieves up to 146\% higher accuracy while using 10$\times$ lower noise scale.
**vs. S\&P [@Ash2020]:** S\&P applies uniform noise; UPGD-W **selectively gates noise injection** based on utility. With the same base Gaussian noise, this selective injection achieves up to 146\% higher accuracy by concentrating perturbations on low-utility parameters.

**vs. Progressive Networks [@Rusu2016]:** Progressive networks freeze previous modules and add new capacity. UPGD-W allows continuous adaptation of all parameters with soft protection for important ones.

**vs. L2 Init [@Kumar2023]:** L2 Init regularizes toward initialization; UPGD-W protects high-utility parameters through reduced updates. Both preserve beneficial initialization properties but through different mechanisms.

### Computational Considerations

UPGD-W adds minimal overhead to standard SGD:

| Component | Cost per Step |
|-----------|---------------|
| Utility computation | $O(N)$ multiplications |
| EMA update | $O(N)$ additions |
| Global max | $O(N)$ reduction |
| Sigmoid scaling | $O(N)$ operations |
| Noise sampling | $O(N)$ random samples |
: UPGD-W computational overhead per step. {#tbl:computational_cost}

Total: $O(N)$ time and memory, identical complexity to gradient computation. For a network with $N$ parameters, UPGD-W roughly doubles per-step compute while providing substantial accuracy improvements.

## Preliminary Reinforcement Learning Evaluation
<!-- SHARE-SKIP-START -->
To test whether our findings generalize beyond supervised continual learning, we conduct preliminary experiments in deep reinforcement learning settings where plasticity loss is a well-documented bottleneck [@Klein2024Survey; @Nikishin2022; @Lyle2023].

### Experimental Setup (RL)

**Environments.** We evaluate on standard RL benchmarks that exhibit non-stationarity:

1. **Atari 2600 games** with periodic resets (following @Nikishin2022): The agent trains continuously, with the environment resetting to force adaptation to changing state distributions.

2. **MuJoCo continuous control** with policy churn: Value-based methods with bootstrapped targets exhibit natural non-stationarity as the policy improves.

<!-- TODO: Add specific environment names and hyperparameters -->

**Architecture.** We use standard architectures: Nature DQN for Atari and MLP critic for MuJoCo, applying UPGD-W variants to different network components.

**Baselines.** We compare UPGD-W (Output Only), UPGD-W (Full), and standard optimizers (Adam, Adam + L2 Init, Adam + S\&P).

### RL Results

<!-- FIGURE PLACEHOLDER: RL training curves -->
<!-- ![Reinforcement learning training curves comparing UPGD-W variants on [Environment Name]. (a) Episode return over training. (b) Value function loss. (c) Dead unit fraction in critic network.](figures/rl/rl_training_curves.png){#fig:rl_training width=90%} -->

**Placeholder for Figure: RL Training Curves.** Training curves comparing UPGD-W variants across RL environments. Expected panels: (a) episodic return, (b) critic loss, (c) dead unit evolution.

<!-- FIGURE PLACEHOLDER: RL layer-selective comparison -->
<!-- ![Layer-selective gating comparison in RL. Comparing output-only vs. full gating in the critic network across different environments.](figures/rl/rl_layer_comparison.png){#fig:rl_layer_comparison width=90%} -->

**Placeholder for Figure: RL Layer-Selective Comparison.** Bar chart comparing layer-selective strategies in RL critic networks.

### Preliminary Observations

**[To be filled with actual results. Expected findings based on theory:]**

1. **Value function as "head".** In actor-critic methods, the value head estimates expected returns conditioned on state features. Under policy improvement, the optimal value function changes (analogous to target shift), suggesting output-only gating on the value head may be beneficial.

2. **Bootstrapping creates non-stationarity.** TD learning with bootstrapped targets induces target shift even in fixed environments. This aligns with our theoretical framework: the "correct answer" (bootstrap target) changes as the network improves.

3. **Replay buffer interactions.** Unlike online supervised learning, RL with replay revisits past experiences. The interaction between utility-based gating and replay sampling merits investigation---high-utility parameters identified on recent experiences may not be optimal for replayed transitions.

### Connection to RL Plasticity Literature

Our preliminary RL results connect to several phenomena documented in the deep RL plasticity literature:

**Primacy bias [@Nikishin2022].** Networks trained with early experiences develop "primacy bias"---difficulty adapting to later experience distributions. UPGD-W's utility-based protection may mitigate this by identifying and allowing adaptation of parameters that become stale.

**Dormant neurons in RL [@Sokar2023].** The dormant neuron phenomenon is particularly severe in RL. Our supervised learning results show UPGD-W maintains lower dead unit fractions; RL experiments will test whether this transfers.

**Regenerative regularization [@JulianiAsh2024OnPolicy].** Recent work shows regenerative (L2-to-init) methods can outperform targeted interventions in on-policy RL. UPGD-W combines selective perturbation with weight decay, potentially capturing benefits of both approaches.

<!-- FIGURE PLACEHOLDER: RL utility dynamics -->
<!-- ![Utility dynamics in RL training. Per-layer utility distributions at different training stages, showing how utility patterns evolve as the policy improves.](figures/rl/rl_utility_dynamics.png){#fig:rl_utility_dynamics width=90%} -->

**Placeholder for Figure: RL Utility Dynamics.** Utility histogram evolution during RL training, comparing early vs. late training phases.

### Limitations of RL Evaluation

This preliminary evaluation has several limitations that warrant future work:

1. **Limited environment coverage.** We test only [N] environments; broader evaluation across environment types (sparse reward, dense reward, continuous, discrete) is needed.

2. **Single-seed results.** RL experiments are notoriously high-variance; multi-seed evaluation with confidence intervals is essential for robust conclusions.

3. **Hyperparameter transfer.** We use supervised-learning-tuned hyperparameters ($\beta$, noise scale); RL-specific tuning may improve results.
3. **Hyperparameter transfer.** We use supervised-learning-tuned hyperparameters (e.g., $\beta$); RL-specific tuning may improve results.

4. **No actor-critic decomposition study.** We apply gating uniformly to the critic; separate treatment of actor and critic networks may be beneficial.

Despite these limitations, preliminary results suggest that the core insight---layer-selective utility gating---may transfer to RL settings, though the optimal configuration likely differs from supervised learning.
<!-- SHARE-SKIP-END -->

## Limitations

1. **Single-seed evaluation.** Results use a single random seed. Multi-seed experiments would strengthen statistical conclusions.

2. **Fully-connected architecture.** We evaluate only MLP architectures. Effectiveness on CNNs and Transformers remains unexplored, though the layer-selective findings suggest promising directions.

3. **Synthetic task boundaries.** Our benchmarks use artificial task boundaries. Real-world continual learning often involves gradual distribution shifts.

4. **Hyperparameter sensitivity.** Optimal $\beta$ varies across datasets (0.9 to 0.9999), requiring task-specific tuning.

5. **Online learning only.** All experiments use batch size 1. Scaling to larger batches typical of modern deep learning is unexplored.

6. **Measurement ambiguity.** Following the survey’s emphasis on evaluation-dependent definitions, our metrics (accuracy, short-horizon plasticity proxy, dead units, norms) capture only a subset of plasticity-related pathologies and do not isolate causality (e.g., dormant units may be a symptom rather than a cause). In particular, “dead units” (zero ReLU activations) are a coarse proxy for reduced capacity; alternative definitions (e.g., normalized dormancy ratios) can behave differently across architectures and regimes [@Klein2024Survey; @Sokar2023]. A more complete analysis would triangulate multiple diagnostics (representation rank, curvature proxies, and generalization-gap style evaluations) to disentangle trainability vs. generalizability effects [@Klein2024Survey; @Berariu2021].

## Future Directions

1. **Architecture-specific gating.** Extending layer-selective insights to attention mechanisms and convolutional kernels.

2. **Extended theoretical analysis.** Propositions A and B provide initial formal grounding for output-only dominance and tail sensitivity. Full dynamic regret bounds, convergence analysis, and connections to curvature preservation remain open for future work.
   
3. **Automatic $\beta$ adaptation.** Detecting distribution shifts and adjusting utility decay could eliminate manual tuning.
   
4. **Large-scale evaluation.** Testing on foundation models and long-horizon RL benchmarks.

5. **Combination with other methods.** Integrating UPGD-W with replay buffers or architectural approaches.

6. **Broader regime testing (RL + realistic shifts).** The Klein survey emphasizes that plasticity loss and the efficacy of regularizers can vary strongly across environments, architectures, and non-stationarity types, motivating broader evaluation to avoid overfitting conclusions to a narrow benchmark family [@Klein2024Survey]. An actionable next step is to evaluate UPGD-style gating in RL training loops where input/target non-stationarity arises naturally (and where replay-based recovery is not always available), and to test whether output-only gating remains optimal under those dynamics.

## Conclusion

We presented a systematic empirical study of Utility-based Perturbed Gradient Descent with weight decay (UPGD-W), analyzing its effectiveness across four continual learning benchmarks and eight method variants.

**Key findings:**

1. **Output-only gating dominates.** Applying utility gating only to the output layer achieves the best performance, with up to 146\% improvement over Shrink \& Perturb on Mini-ImageNet. This reveals that hidden layers benefit from unrestricted plasticity while output layers require selective protection.

2. **Natural utility dynamics are essential.** Constraining utility values to narrow ranges degrades performance by up to 51\%, demonstrating that the gradient-based utility estimation produces effective importance patterns that should not be artificially constrained.

3. **First-order utility is sufficient.** UPGD-W achieves strong results using only gradient-weight products, suggesting that Hessian-based second-order computations are unnecessary for effective parameter importance estimation.

4. **Selective perturbation preserves capacity.** UPGD-W maintains lower dead unit fractions than S\&P, indicating better preservation of network capacity through selective rather than uniform noise injection.

5. **Layer-specific treatment has theoretical grounding.** Propositions A and B provide formal motivation for why the utility tails drive performance through tail-based gating (A) and why output-only gating succeeds under target shift (B).

These findings establish clear design principles for utility-based continual learning optimizers: focus gating on output layers, allow natural utility dynamics, and preserve hidden layer plasticity. We release our implementation to facilitate future research in this direction.

**Code availability:** Implementation available at `https://github.com/bnshin83/upgd.git`

\newpage
\appendix
\onecolumn

## Theoretical Proofs
<!-- SHARE-SKIP-START: label -->
This appendix provides formal proofs for the two propositions stated in Section 3.7 (Theoretical Motivation). We first establish notation, then prove each proposition with explicit assumptions and quantitative bounds.

### Notation and Setup

Consider a neural network decomposed as representation plus head:
$$f(x; \theta) = f(x; V, W) = W \phi(x; V)$$
where $V \in \mathbb{R}^{d_{V}}$ parameterizes hidden layers and $W \in \mathbb{R}^{d_{W}}$ the output layer. Let $\theta = (V, W) \in \mathbb{R}^N$ with $N = d_{V} + d_{W}$.

**Definition 1 (UPGD-W Update).** The UPGD-W update for parameter $\theta_{i}$ at time $t$ is:
$$\theta_{t+1,i} = (1 - \eta\lambda)\theta_{t,i} - \eta\,a_{t,i}(g_{t,i} + \varepsilon_{t,i})$$
where $g_{t,i} = \partial \ell_t / \partial \theta_i$ is the gradient, $\varepsilon_{t,i} \sim \mathcal{N}(0,1)$ is independent injected noise (fixed standard normal), and $\lambda \geq 0$ is weight decay.

**Definition 2 (Utility and Gating).** The raw utility is $\tilde{u}_{t,i} = -g_{t,i} \cdot \theta_{t,i}$. The bias-corrected utility uses exponential moving average:
$$\hat{u}_{t,i} = \frac{\bar{u}_{t,i}}{1 - \beta^t}, \quad \bar{u}_{t,i} = \beta \bar{u}_{t-1,i} + (1-\beta)\tilde{u}_{t,i}$$
The gate is formed via global-max normalization and sigmoid:
$$s_{t,i} = \sigma\left(\frac{\hat{u}_{t,i}}{\max(u_{\max,t}, \epsilon)}\right), \quad u_{\max,t} = \max_{j \in [N]} \hat{u}_{t,j}, \quad a_{t,i} = 1 - s_{t,i}$$
where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic sigmoid and $\epsilon > 0$ is a small numerical-stability constant.

**Definition 3 (Parameter Partition).** For threshold $\tau > 0$, partition parameters into:
\begin{align}
\mathcal{I}^{+}_{t}(\tau) &= \{i : s_{t,i} \geq 0.5 + \tau\} \quad \text{(high-utility, protected)} \\
\mathcal{I}^{-}_{t}(\tau) &= \{i : s_{t,i} \leq 0.5 - \tau\} \quad \text{(low-utility, exploratory)} \\
\mathcal{B}_{t}(\tau) &= \{i : |s_{t,i} - 0.5| < \tau\} \quad \text{(bulk, neutral)}
\end{align}

### Proof of Proposition A (Effective Tail-Based Gating)

**Proposition A (Formal Statement).** *Let $\tau \in (0, 0.5)$ and suppose the utility distribution satisfies the concentration condition: $|\mathcal{B}_{t}(\tau)| \geq (1 - \delta)N$ for some $\delta \in (0, 1)$, where $N = \dim(\theta)$ and $\delta$ upper bounds the fraction of parameters outside the bulk set $\mathcal{B}_t(\tau)$ (i.e., in $\mathcal{I}^+_t(\tau)\cup \mathcal{I}^-_t(\tau)$). Then the UPGD-W update admits the decomposition:*
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \eta \cdot \frac{1}{2}(g_t + \varepsilon_t) + \eta \cdot \Delta_t$$
*where the deviation term is $\Delta_t = -\text{diag}(a_t - 0.5 \cdot \mathbf{1})(g_t + \varepsilon_t)$ and satisfies $\|\Delta_t\|_\infty \leq \frac{1}{2}\|g_t + \varepsilon_t\|_\infty$. Moreover, $|\mathcal{I}^+_t(\tau)| + |\mathcal{I}^-_t(\tau)| \leq \delta N$, and for any $i \in \mathcal{B}_t(\tau)$ we have $|[\Delta_t]_i| < \tau\,|g_{t,i} + \varepsilon_{t,i}|$.*

**Lemma A.1 (Sigmoid Bounds).** For $x \in [0, 1]$:
\begin{align}
\sigma(x) &\in [\sigma(0), \sigma(1)] = [0.5, 0.731] \\
|\sigma(x) - 0.5| &\leq \sigma'(0) \cdot |x| = 0.25|x| \quad \text{for } x \text{ near } 0
\end{align}

*Proof.* The sigmoid $\sigma(x) = 1/(1+e^{-x})$ is monotonically increasing with $\sigma(0) = 0.5$ and $\sigma(1) \approx 0.731$. By Taylor expansion around $x=0$: $\sigma(x) = 0.5 + 0.25x + O(x^3)$. $\square$



**Lemma A.2 (Ratio Concentration).** Define the normalized ratio $r_{t,i} = \hat{u}_{t,i}/u_{\max,t} \in [0,1]$. If utilities have mean $\mu$ and the maximum satisfies $u_{\max,t} \geq \mu + \gamma$ for some gap $\gamma > 0$, then for parameter $i$ with $\hat{u}_{t,i} \leq \mu + \gamma/2$:
$$r_{t,i} \leq \frac{\mu + \gamma/2}{\mu + \gamma} = 1 - \frac{\gamma/2}{\mu + \gamma} < 1$$

*Proof.* Direct computation from the ratio definition. $\square$

*Remark (Not required for Proposition A).* Lemma A.2 provides intuition for why global-max normalization can yield many ratios $r_{t,i}$ well below 1; it is not used in the proof of Proposition A below, which relies only on the bulk/tail partition and the concentration condition.

**Lemma A.3 (Gate Deviation Bound).** For any $i \in \mathcal{B}_t(\tau)$, the gate deviation from neutral satisfies:
$$|a_{t,i} - 0.5| = |s_{t,i} - 0.5| < \tau$$

*Proof.* By definition of $\mathcal{B}_t(\tau)$, $|s_{t,i} - 0.5| < \tau$. Since $a_{t,i} = 1 - s_{t,i}$, we have $a_{t,i} - 0.5 = 0.5 - s_{t,i}$, so $|a_{t,i} - 0.5| = |s_{t,i} - 0.5| < \tau$. $\square$

**Proof of Proposition A.**

*Step 1: Exact Update Decomposition.*
Write the UPGD-W update component-wise:
$$\theta_{t+1,i} = (1-\eta\lambda)\theta_{t,i} - \eta a_{t,i}(g_{t,i} + \varepsilon_{t,i})$$

Decompose $a_{t,i} = 0.5 + (a_{t,i} - 0.5)$:
\begin{align}
\theta_{t+1,i} &= (1-\eta\lambda)\theta_{t,i} - \eta \cdot 0.5 \cdot (g_{t,i} + \varepsilon_{t,i}) - \eta(a_{t,i} - 0.5)(g_{t,i} + \varepsilon_{t,i})
\end{align}

In vector form:
$$\theta_{t+1} = (1-\eta\lambda)\theta_t - \frac{\eta}{2}(g_t + \varepsilon_t) - \eta \cdot \text{diag}(a_t - 0.5 \cdot \mathbf{1})(g_t + \varepsilon_t)$$

*Step 2: Deviation Term Analysis.*
Define $\Delta_t = -\text{diag}(a_t - 0.5 \cdot \mathbf{1})(g_t + \varepsilon_t)$, so component-wise:
$$[\Delta_t]_i = -(a_{t,i} - 0.5)(g_{t,i} + \varepsilon_{t,i})$$

For $i \in \mathcal{B}_t(\tau)$: By Lemma A.3, $|a_{t,i} - 0.5| < \tau$, so:
$$|[\Delta_t]_i| < \tau |g_{t,i} + \varepsilon_{t,i}|$$

For $i \in \mathcal{I}^+_t(\tau)$: $a_{t,i} < 0.5 - \tau$, so $(a_{t,i} - 0.5) < -\tau < 0$ (protection).

For $i \in \mathcal{I}^-_t(\tau)$: $a_{t,i} > 0.5 + \tau$, so $(a_{t,i} - 0.5) > \tau > 0$ (exploration).

*Step 3: Sparsity Bound.*
By the concentration condition, $|\mathcal{B}_t(\tau)| \geq (1-\delta)N$, so:
$$|\mathcal{I}^+_t(\tau)| + |\mathcal{I}^-_t(\tau)| \leq \delta N$$

Therefore at most $\delta N$ components fall in the tail sets $\mathcal{I}^+_t(\tau)\cup \mathcal{I}^-_t(\tau)$, i.e., at most $\delta N$ coordinates have $|a_{t,i} - 0.5| \geq \tau$.

*Step 4: Magnitude Bound.*
Since $a_{t,i} \in (0, 1)$, we have $|a_{t,i} - 0.5| < 0.5$, giving:
$$\|\Delta_t\|_\infty < \frac{1}{2}\|g_t + \varepsilon_t\|_\infty$$

**Corollary A.4 (Effective Learning Rate Modulation).** The effective learning rate for parameter $i$ is $\eta_{\text{eff},i} = \eta \cdot a_{t,i}$. For the three regimes:
\begin{align}
i \in \mathcal{I}^+_t(\tau): \quad &\eta_{\text{eff},i} < \eta(0.5 - \tau) \quad \text{(reduced, protection)} \\
i \in \mathcal{B}_t(\tau): \quad &\eta_{\text{eff},i} \in (\eta(0.5-\tau), \eta(0.5+\tau)) \quad \text{(near-neutral)} \\
i \in \mathcal{I}^-_t(\tau): \quad &\eta_{\text{eff},i} > \eta(0.5 + \tau) \quad \text{(amplified, exploration)}
\end{align}

**Empirical Verification.** With $\tau = 0.02$ (defining the $[0.48, 0.52]$ bulk region), our experiments show $\delta \leq 0.003$ across all datasets, validating the concentration condition. The partition satisfies:
- $|\mathcal{B}_t(0.02)|/N \geq 0.997$ (bulk)
- $|\mathcal{I}^+_t(0.02)|/N \leq 0.002$ (protected)
- $|\mathcal{I}^-_t(0.02)|/N \leq 0.001$ (exploratory)

This confirms Proposition A: UPGD-W differs from baseline noisy SGD (with step $\eta/2$) by a $\tau$-scaled modulation on the bulk, and by potentially larger modulation on a sparse ($<0.3\%$) set of extreme-utility parameters. $\square$

---

### Proof of Proposition B (Output-Only Advantage)

**Proposition B (Formal Statement).** *Consider a target-shift stream satisfying assumptions (B1)-(B4) below. Let $\overline{R}^{\text{full}}_T$ and $\overline{R}^{\text{out}}_T$ denote the dynamic regret of full gating and output-only gating, respectively. Then:*
$$\overline{R}^{\text{out}}_T \leq \overline{R}^{\text{full}}_T - \Delta_{\text{feature}}(T)$$
*where $\overline{R}^{\text{out}}_T$ and $\overline{R}^{\text{full}}_T$ are dynamic-regret upper bounds for output-only and full gating, respectively. Here $\Delta_{\text{feature}}(T) \geq 0$ is the cumulative feature degradation penalty, with $\Delta_{\text{feature}}(T) > 0$ when full gating constrains representation adaptation.*

**Assumptions.**

**(B1) Piecewise-stationary tasks.** The data stream consists of $K$ tasks, task $k$ lasting $U_k$ steps with distribution $\mathcal{D}_k$. Total horizon $T = \sum_{k=1}^K U_k$.

**(B2) Target-shift structure.** Under label permutation:
- **Feature stability:** $\|\mathbb{E}_{\mathcal{D}_{k}}[\phi(x; V^*)] - \mathbb{E}_{\mathcal{D}_{k+1}}[\phi(x; V^*)]\| \leq \epsilon_V$ for small $\epsilon_V \geq 0$. Motivated by evidence that shallow layers learn general features [@Yosinski2014; @Wei2024].
- **Head drift:** $\|W^*_{k+1} - W^*_{k}\| \geq \rho$ for some $\rho > 0$ at each task boundary.

**(B3) Smooth convex head loss.** The loss $\ell_t(W) = \ell(W\phi(x_t; V_t), y_t)$ is $\mu$-strongly convex and $L$-smooth in $W$:
$$\frac{\mu}{2}\|W - W'\|^2 \leq \ell_t(W) - \ell_t(W') - \langle \nabla\ell_t(W'), W-W'\rangle \leq \frac{L}{2}\|W - W'\|^2$$

**(B4) Bounded gradients.** $\|\nabla_W \ell_t\| \leq G_W$ and $\|\nabla_V \ell_t\| \leq G_V$ for all $t$.

**(B5) Lipschitz feature dependence.** The head gradient depends on features: $\|\nabla_W\ell_t(W; \phi) - \nabla_W\ell_t(W; \phi')\| \leq L_\phi \|\phi - \phi'\|$.

**Definition 4 (Dynamic Regret).** For comparator sequence $\{W^*_t\}_{t=1}^T$:
$$\text{DynReg}_T(W) = \sum_{t=1}^T \ell_t(W_t) - \sum_{t=1}^T \ell_t(W^*_t)$$

**Definition 5 (Path Length).** The path length of the comparator is:
$$P_T = \sum_{t=1}^{T-1} \|W^*_{t+1} - W^*_t\|$$

Under (B2), $P_T \geq (K-1)\rho$ where $K$ is the number of tasks.

**Lemma B.1 (Dynamic Regret Bound for Preconditioned GD).** Consider updates $W_{t+1} = W_t - \eta A_t \nabla\ell_t(W_t)$ where $A_t = \text{diag}(a_{t,1}, \ldots, a_{t,d_W})$ with $a_{t,i} \in [a_{\min}, a_{\max}]$. Under (B3)-(B4):
$$\text{DynReg}_T \leq \frac{\|W_1 - W^*_1\|^2}{2\eta a_{\min}} + \frac{\eta a_{\max} G_W^2 T}{2} + \frac{D\,P_T}{\eta a_{\min}}$$

*Proof.* Standard analysis of online gradient descent with preconditioner [@Zinkevich2003; @Hazan2016]. The key terms are: (i) initial distance, (ii) cumulative gradient contribution, (iii) comparator drift. The preconditioner $A_t$ scales the effective step size, appearing in the denominator for tracking terms. $\square$

**Lemma B.2 (Feature Quality Impact).** Let $\phi^{\text{full}}_t$ and $\phi^{\text{out}}_t$ denote features under full gating and output-only gating, respectively. Under (B5), the gradient quality difference is:
$$\|\nabla_W\ell_t(W; \phi^{\text{full}}_t) - \nabla_W\ell_t(W; \phi^{\text{out}}_t)\| \leq L_\phi \|\phi^{\text{full}}_t - \phi^{\text{out}}_t\|$$

**Lemma B.3 (Feature Adaptation Under Gating).** Consider hidden layer updates:
\begin{align}
\text{Full gating:} \quad V^{\text{full}}_{t+1} &= V^{\text{full}}_t - \eta A^V_t (\nabla_V\ell_t + \varepsilon_t) \\
\text{Output-only:} \quad V^{\text{out}}_{t+1} &= V^{\text{out}}_t - \frac{\eta}{2}(\nabla_V\ell_t + \varepsilon_t)
\end{align}
where $A^V_t$ has some entries near zero (for high-utility hidden parameters). Starting from $V_1^{\text{full}} = V_1^{\text{out}}$, after $T$ steps:
$$\mathbb{E}\|\phi(x; V^{\text{full}}_T) - \phi(x; V^{\text{out}}_T)\|^2 \geq \Omega\left(\eta^2 T \cdot \sum_{i \in \mathcal{I}^+_V} (0.5 - a^V_{t,i})^2 G_V^2\right)$$
where $\mathcal{I}^+_V$ is the protected set in hidden layers.

*Proof sketch.* The difference accumulates from differential updates on protected coordinates. Each step contributes $(0.5 - a^V_{t,i})\eta g_{V,i}$ to the gap for protected coordinate $i$. Over $T$ steps, this gap grows with the accumulated squared difference. $\square$

**Proof of Proposition B.**

*Step 1: Apply Lemma B.1 to both strategies.*

For output-only gating, hidden layers have uniform $a^V_i = 0.5$, and head gating operates on $W$:
$$\overline{R}^{\text{out}}_T \leq \frac{D_0^2}{2\eta a^W_{\min}} + \frac{\eta a^W_{\max} G_W^2 T}{2} + \frac{D\,P_T}{\eta a^W_{\min}}$$
where $a^W_{\min}, a^W_{\max}$ are the gate bounds on head parameters.

For full gating, both $V$ and $W$ are gated:
$$\overline{R}^{\text{full}}_T \leq \frac{D_0^2}{2\eta a_{\min}} + \frac{\eta a_{\max} \tilde{G}_W^2 T}{2} + \frac{D\,P_T}{\eta a_{\min}}$$
where $\tilde{G}_W = G_W + L_\phi \|\phi^{\text{full}} - \phi^*\|$ reflects degraded feature quality.

*Step 2: Quantify feature degradation penalty.*

By Lemma B.3, full gating causes feature divergence from the optimal adaptation path. Define:
$$\Delta_{\text{feature}}(T) = \frac{\eta T}{2}\left(\tilde{G}_W^2 - G_W^2\right) = \frac{\eta T}{2}\left(2G_W L_\phi \|\phi^{\text{full}} - \phi^*\| + L_\phi^2\|\phi^{\text{full}} - \phi^*\|^2\right)$$

This is positive when $\|\phi^{\text{full}} - \phi^*\| > 0$, i.e., when full gating prevents optimal feature adaptation.

*Step 3: Combine bounds.*

Under target shift (B2), the path length $P_T$ is dominated by head drift: $P_T \geq (K-1)\rho$. The tracking term $D\,P_T/(\eta a_{\min})$ requires sufficient effective step size $\eta a_{\min}$ to keep regret bounded.

Full gating reduces $a_{\min}$ on head coordinates that develop high utility, *and* degrades feature quality. Output-only gating:
- Maintains $a^V_i = 0.5$ uniformly (no feature degradation)
- Still allows selective gating on $W$ for stability-plasticity tradeoff

Therefore:
$$\overline{R}^{\text{out}}_T \leq \overline{R}^{\text{full}}_T - \Delta_{\text{feature}}(T)$$

with equality when hidden layer gating has no effect (i.e., no high-utility hidden parameters).

**Corollary B.4 (Conditions for Output-Only Advantage).** Output-only gating strictly dominates ($\Delta_{\text{feature}} > 0$) when:
1. Hidden layers develop high-utility parameters under full gating ($|\mathcal{I}^+_V| > 0$)
2. Feature quality affects head optimization ($L_\phi > 0$)
3. Sufficient training time for gap accumulation ($T$ large)

**Empirical Support.** Our per-layer analysis confirms:
- Output layer: 2--7$\times$ larger utility maxima than hidden layers
- Output layer: 2--98$\times$ more mass in high-utility tail
- Input-MNIST (input shift): minimal layer-selective difference, consistent with $\epsilon_V$ being large

This validates that under target shift, utility concentrates in the head, making output-only gating optimal. Under input shift, the assumption (B2) fails ($\epsilon_V$ large), and the theory predicts no clear winner---consistent with our empirical observations. $\square$

---

### Implication: Why Clamping Hurts Performance

**Corollary C.1 (Clamping Degrades Selectivity).** *Let $s^{\text{clamp}}_{t,i} = \text{clip}(s_{t,i}, s_{\min}, s_{\max})$ for bounds $0 < s_{\min} < 0.5 < s_{\max} < 1$. Define the clamped gate $a^{\text{clamp}}_{t,i} = 1 - s^{\text{clamp}}_{t,i}$. Then:*

1. *Gate range compression:* $a^{\text{clamp}}_{t,i} \in [1-s_{\max}, 1-s_{\min}]$, a strict subset of $(0,1)$
2. *Tail elimination:* For $\tau > s_{\max} - 0.5$, the protected set $\mathcal{I}^+_t(\tau) = \emptyset$
3. *Variance collapse:* $\text{Var}(a^{\text{clamp}}_t) \leq \text{Var}(a_t)$ with equality iff no clamping occurs

**Proof.**

*Part 1: Gate range compression.*
By definition, $s^{\text{clamp}}_{t,i} \in [s_{\min}, s_{\max}]$, so:
$$a^{\text{clamp}}_{t,i} = 1 - s^{\text{clamp}}_{t,i} \in [1 - s_{\max}, 1 - s_{\min}]$$

For example, with $[s_{\min}, s_{\max}] = [0.48, 0.52]$:
$$a^{\text{clamp}}_{t,i} \in [0.48, 0.52]$$
eliminating both strong protection ($a \to 0$) and strong exploration ($a \to 1$).

*Part 2: Tail elimination.*
The protected set requires $s_{t,i} \geq 0.5 + \tau$. Under clamping with $s_{\max} < 0.5 + \tau$:
$$s^{\text{clamp}}_{t,i} \leq s_{\max} < 0.5 + \tau \implies \mathcal{I}^+_t(\tau) = \emptyset$$

Similarly for the exploratory set when $s_{\min} > 0.5 - \tau$.

*Part 3: Variance collapse.*
Clamping is a contraction mapping that reduces spread. Formally:
$$\text{Var}(a^{\text{clamp}}) = \mathbb{E}[(a^{\text{clamp}} - \bar{a}^{\text{clamp}})^2] \leq \mathbb{E}[(a - \bar{a})^2] = \text{Var}(a)$$
with strict inequality when any $a_{t,i}$ falls outside $[1-s_{\max}, 1-s_{\min}]$. $\square$

**Theorem C.2 (Clamping Reduces to Constant-Step SGD).** *Under tight clamping with $s_{\max} - s_{\min} \to 0$, UPGD-W converges to constant-step noisy SGD:*
$$\lim_{s_{\max} - s_{\min} \to 0} \theta^{\text{clamp}}_{t+1} = (1-\eta\lambda)\theta_t - \eta \bar{a}(g_t + \varepsilon_t)$$
*where $\bar{a} = (1 - s_{\min} + 1 - s_{\max})/2 = 1 - (s_{\min} + s_{\max})/2$.*

**Proof.** As $s_{\max} - s_{\min} \to 0$, all gates converge to $\bar{a}$:
$$a^{\text{clamp}}_{t,i} \to \bar{a} \quad \forall i$$

The update becomes:
$$\theta^{\text{clamp}}_{t+1,i} = (1-\eta\lambda)\theta_{t,i} - \eta \bar{a}(g_{t,i} + \varepsilon_{t,i})$$

This is standard noisy SGD with effective step size $\eta\bar{a}$ and injected noise standard deviation scaled by $\bar{a}$. The utility-based selectivity is completely eliminated. $\square$

**Quantitative Impact.** Let $\alpha_{\text{eff}} = |\mathcal{I}^+_t| + |\mathcal{I}^-_t|$ be the effective tail mass (fraction of selectively treated parameters). Under clamping to $[0.48, 0.52]$:
$$\alpha_{\text{eff}}^{\text{clamp}} = 0 \quad \text{vs.} \quad \alpha_{\text{eff}}^{\text{natural}} \approx 0.003$$

The empirical performance degradation of up to 51\% suggests that even this small 0.3\% selective treatment accounts for a substantial portion of UPGD-W's advantage.

**Asymmetric Clamping Analysis.** Our experiments show asymmetric effects:
- Clamping to $[0, 0.52]$ (preserving low tail): moderate degradation
- Clamping to $[0.48, 1]$ (preserving high tail): moderate degradation
- Clamping to $[0.48, 0.52]$ (eliminating both): severe degradation
- Clamping to $[0.44, 0.56]$ (symmetric widening): still significant degradation

This pattern suggests both tails contribute, with their joint elimination being most harmful. The low-utility tail enables exploration of underutilized parameters, while the high-utility tail provides protection---both mechanisms are necessary for optimal continual learning. $\square$

---

### Connection to Curvature-Based Plasticity Theory

Our utility metric $\tilde{u}_{t,i} = -g_{t,i} \cdot \theta_{t,i}$ can be interpreted through the lens of @Lewandowski2024's curvature-based explanation of plasticity loss.

**Observation.** Under certain conditions, high utility correlates with sensitivity to perturbation (curvature). Specifically, if:
1. The loss surface is approximately quadratic near $\theta_{t}$: $\ell(\theta) \approx \ell(\theta_{t}) + g_{t}^\top(\theta - \theta_{t}) + \frac{1}{2}(\theta - \theta_{t})^\top H_{t}(\theta - \theta_{t})$
2. The parameter direction $e_{i}$ has significant curvature: $[H_{t}]_{ii} > 0$

Then perturbing parameter $i$ by $\delta$ incurs loss change $\approx \frac{1}{2}[H_{t}]_{ii}\delta^2$. Parameters with high utility (contributing to loss reduction) often lie along directions with non-negligible curvature---precisely the directions that @Lewandowski2024 argue must be preserved for plasticity.

**Conjecture.** UPGD-W's utility-gated protection preserves curvature directions implicitly by protecting parameters whose current values are actively contributing to loss minimization. A formal proof would require explicit curvature measurements, which we leave to future work.

---

<!-- SHARE-SKIP-END -->

### Additional Empirical Details

**Per-Layer Utility Statistics (Last 20% of Training, Seed=2).**

| Dataset | Layer | Tail Mass (>0.52) | Raw $u_{\max}$ | Output/Hidden Ratio |
|---------|-------|-------------------|----------------|---------------------|
| CIFAR-10 | linear\_{3} | 1.06--1.08% | 0.42--0.51 | 3.8--5.0$\times$ |
| CIFAR-10 | hidden avg | 0.22--0.28% | 0.21--0.30 | -- |
| EMNIST | linear\_{3} | 0.24--0.39% | 0.38--0.52 | 2.3--5.4$\times$ |
| EMNIST | hidden avg | 0.07--0.11% | 0.09--0.15 | -- |
| Mini-ImageNet | linear\_{3} | 0.31--0.42% | 0.29--0.48 | 1.3--2.1$\times$ |
| Mini-ImageNet | hidden avg | 0.20--0.24% | 0.07--0.13 | -- |
| Input-MNIST | linear\_{3} | 17.7--18.6% | 0.47--0.52 | 76--98$\times$ |
| Input-MNIST | hidden avg | 0.19--0.24% | 0.15--0.18 | -- |
: Per-layer utility statistics across datasets. {#tbl:per_layer_utility}

These statistics confirm that the output layer concentrates both the extreme utility tail mass and the largest utility maxima, supporting the head-centric non-stationarity picture under target shift.

### Reconciling Conflicting Evidence on Memorization Localization

Recent work on memorization under label noise appears to disagree on *where* memorization "lives" in deep networks. A useful way to reconcile these claims is to distinguish (i) the *question being asked* and (ii) the *experimental regime*. One line of evidence (e.g., Anagnostidis et al., 2023) reports sharp layer-wise transitions in *decodability* under strong data augmentation and an explicit encoder--projector architecture: noisy-label/instance information becomes much more easily recoverable from late representations (often at the projector boundary), while earlier layers retain features that probe well on clean structure. Another line of evidence (e.g., Maini et al., 2023) emphasizes that memorized predictions can be supported by *distributed* sets of neurons/parameters across many layers, based on intervention-style analyses (layer rewinding/retraining and neuron-level "criticality").

These results need not be contradictory because they target different notions of localization. **Probe-based localization** asks where a simple decoder (e.g., a k-NN probe) can recover information; **intervention-based localization** asks which parameters are causally sufficient/necessary for a behavior. Probe localization can be sharply late even when causal support is distributed, since distributed changes can "funnel" into a representation that becomes linearly separable only late. Conversely, interventions can reveal redundancy even when information is not uniformly decodable across layers.

Any unified claim is conditional on training details. Strong augmentation can impose invariance constraints and capacity pressure that make instance-specific fitting harder to express uniformly across the stack, while projector-style heads provide a natural late module where task- or instance-specific information can concentrate. By contrast, in standard supervised classifiers without such architectural separation, optimization may realize memorized exceptions via many small, distributed adjustments. Noise rate also changes the regime: extreme random-label settings probe "memorization in the absence of task signal," whereas partial-noise settings probe "memorizing exceptions on top of a learnable pattern."

**Connection to our setting.** Our experiments do not directly study random-label memorization. Instead, label-permuted benchmarks induce repeated *target shifts*, where rapid remapping in the output layer is central. In this regime, our finding that **output-only utility gating dominates** supports a more modest claim: the head is a critical locus for managing adaptation pressure under target shift, even if representation layers may still participate in (and be necessary for) learning dynamics. Concretely, our per-layer utility logs show that the **extreme-utility tail** is disproportionately concentrated in the output layer, consistent with a "late module" concentrating task-specific signal in the sense of probe-based localization (Anagnostidis et al.). This perspective motivates layer-conditional diagnostics: curvature/plasticity proxies may need to be measured at the head versus throughout the representation stack depending on whether non-stationarity is dominated by target shifts (head-centric) or input shifts / augmentation regimes (more distributed).



<!-- SHARE-SKIP-START -->
## Re-explain (short + intuitive)

### What the “proof sketch” is trying to justify
It’s arguing why **output-only gating** (gating just the head \(W\)) can be better than **full gating** (gating both head \(W\) and representation \(V\)) when the stream is **target-shift / label-permutation**.

### 1) The update is just SGD with a coordinate-wise “dimmer switch”
They write the head update as
\[
W_{t+1} = W_t - \eta\,A_t\big(\nabla \ell_t(W_t) + \sigma \varepsilon_t\big),\quad A_t=\operatorname{diag}(a_{t,i}),\ a_{t,i}\in(0,1).
\]

Interpretation:
- \(a_{t,i}\) scales how much you move in coordinate \(i\).
- If some important coordinates have small \(a_{t,i}\), the optimizer **moves slowly** there.

### 2) Under target shift, the “best head” keeps moving
If labels permute across tasks, the optimal head \(W_t^*\) changes a lot. They summarize this by the path length
\[
P_T=\sum_t \lVert W_{t+1}^* - W_t^* \rVert,
\]
which is **large** under label permutations.

So the optimizer must **track a moving target**.

### 3) Tracking a moving target needs enough effective step size
Dynamic regret bounds typically get worse when:
- the target moves a lot (large \(P_T\)), and
- your step size is effectively too small.

 With gating, the smallest gate value \(a_{\min}\) acts like a worst-case bottleneck, which is why they say the regret scales like
 \[
 O\!\left(\frac{D\,P_T}{\eta a_{\min}}\right),
 \]
 where \(D\) is a diameter bound on the iterates/comparators. Meaning: if gating makes \(a_{\min}\) tiny on critical head coordinates, tracking becomes harder.

### 4) Why full gating can be worse than output-only gating
Full gating does two bad things at once:

- **(A) It can shrink head updates** (via small \(a_{\min}\) on head coordinates), making it harder to follow \(W_t^*\).
- **(B) It can “freeze” the representation \(V\)**, so features \(\phi(x;V)\) can’t adapt. Then the head sees worse features, making head optimization harder. They encode this as an inflated effective gradient bound:
\[
\tilde{G}_W = G_W + L_\phi \lVert \phi^{\text{full}} - \phi^* \rVert,
\]
and the extra cost over time as
\[
\Delta_{\text{feature}}(T)=\frac{\eta T}{2}\left(\tilde{G}_W^2 - G_W^2\right)>0.
\]

### 5) The punchline
Output-only gating avoids (B): it does **not** restrict \(V\) much, so features stay good, and the head’s tracking problem stays “easy”.  
Therefore, in head-dominated nonstationarity, output-only can have strictly lower regret than full gating (by roughly the feature-degradation penalty \(\Delta_{\text{feature}}(T)\)).

## Status
- **Re-explained** the block with the core intuition: gating shrinks effective step sizes; target shift requires tracking; full gating also harms features, so output-only can win.

## Why does the dynamic-regret bound scale like \(1/(\eta a_{\min})\) (and where the \(P_T^2\) can come from)

### 1) Key idea: gating changes the **effective step size**
Your head update is
\[
W_{t+1} = W_t - \eta\,A_t\big(\nabla \ell_t(W_t)+\varepsilon_t\big),\quad A_t=\operatorname{diag}(a_{t,i}).
\]
If we look at worst-case progress across coordinates, the smallest gate
\[
a_{\min} := \min_{t,i} a_{t,i}
\]
acts like a lower bound on how large your coordinate-wise steps can be. So in “worst-case” regret analyses, you can think of the algorithm as having an **effective learning rate**
\[
\eta' \approx \eta a_{\min}.
\]

That’s the origin of the \(1/(\eta a_{\min})\) dependence: smaller gates \(\Rightarrow\) smaller effective steps \(\Rightarrow\) harder to track a moving optimum.

---

### 2) Standard dynamic regret bound template (what Zinkevich-style results look like)
For online gradient descent tracking a *drifting* comparator sequence \(u_t\) (here \(u_t=W_t^*\)), a common form is:
\[
R_T(u_{1:T}) \;\lesssim\; \frac{D^2}{2\eta'} \;+\; \frac{\eta'}{2}\sum_{t=1}^T \|g_t\|^2 \;+\; \frac{D}{\eta'}\,P_T,
\]
where
- \(D\) is a diameter bound (how far iterates/comparators can be),
- \(g_t\) is the gradient (bounded by \(G\) typically),
- \(P_T := \sum_t \|u_{t+1}-u_t\|\) is the **path length**.

If you substitute \(\eta'=\eta a_{\min}\), the drift term becomes
\[
\frac{D}{\eta'}P_T \;=\; \frac{D\,P_T}{\eta a_{\min}}.
\]
So **at minimum**, you should expect scaling like
\[
R_T \propto \frac{P_T}{\eta a_{\min}}
\quad\text{(up to constants and a factor like \(D\)).}
\]

---

### 3) So why did your text have \(P_T^2/(\eta a_{\min})\)?
That \(P_T^2\) is usually a **coarse simplification** that can appear if you upper bound \(D\) by something proportional to \(P_T\).

For example, if the comparator can drift over a region whose diameter is itself on the order of the total movement, you might use a rough bound like
\[
D \;\lesssim\; P_T,
\]
which turns
\[
\frac{D\,P_T}{\eta a_{\min}} \;\lesssim\; \frac{P_T^2}{\eta a_{\min}}.
\]

So the squared dependence is not “the only correct theorem”; it’s a **regime-dependent / loose** way to emphasize:

- **Bigger drift** (\(P_T\) large) hurts tracking a lot.
- **Smaller effective step size** (\(\eta a_{\min}\) small) hurts tracking a lot.

---

### 4) Practical takeaway (the intuition you want in the paper)
- Target shift \(\Rightarrow P_T\) is large (head optimum moves a lot).
- Full gating can shrink \(a_{\min}\) for important head coordinates \(\Rightarrow\) \(\eta a_{\min}\) becomes small.
- Therefore the head can’t track the moving optimum well, and the dynamic regret bound deteriorates roughly like \(1/(\eta a_{\min})\) (and potentially like \(P_T^2\) under coarse bounding).

If you want, I can help you rewrite that sentence in the paper to be more formally accurate (e.g., use \(O(DP_T/(\eta a_{\min}))\) and then note “in settings where \(D \sim P_T\), this becomes \(O(P_T^2/(\eta a_{\min}))\)”).
<!-- SHARE-SKIP-END -->