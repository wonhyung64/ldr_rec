# Review by tCJC

### (Weakness 1) Why does the pure debiased score (\eta=0 in Figure 4) collapse to ~0.00 for BSARec and MF on MovieLens-1M, and how does \eta=0 perform on KuaiRand's unbiased random-exposure subset?

1. By Eq. (3), observed interactions follow $p_0=\pi_0\exp f^*$. Thus, $\eta=0$ estimates a density ratio rather than the observed interaction probability, explaining its generally lower Recall across backbones, not just BSARec/MF. Reintroducing popularity at inference is therefore theoretically motivated: $\eta=0.5$ recovers the ranking of $p_0$ under exact estimation. Consistently, Fig. 4 generally peaks near $\eta=0.3$–$0.5$, outperforming popularity-only scoring ($\eta=1$). KuaiRand's test set already uses random exposure, which does not eliminate popularity-dependent user choices. Hence, low Recall at $\eta=0$ does not imply unsuccessful debiasing.


### (Weakness 2) Why does the Hawkes estimator trail Static/Sliding-window estimators on 9 of 12 cases across MovieLens-1M and KuaiRand (Table 3) and cause BSARec to drop from 0.0771 to 0.0335 on MovieLens-1M?

- **W2.** Hawkes is best in 5/6 Micro-Video settings but only 3/12 settings elsewhere. Its benefit is dataset-dependent; we do not claim universal superiority. The BSARec failure is addressed in our reply to R3.

### (Weakness 3) From your existing Table 1 logs, what is the exact Head and Tail Recall@10 of the Vanilla Backbones compared to Backbone+(Ours)?




| **Model** | **Micro Video** | | **MovieLens** | | **KuaiRand** | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Head** | **Tail** | **Head** | **Tail** | **Head** | **Tail** |
| SASRec | 0.174±0.006 | 0.124±0.002 | 0.106±0.002 | 0.087±0.002 | 0.020±0.002 | 0.027±0.001 |
| +Ours | **0.196±0.004** | **0.147±0.003** | **0.132±0.003** | **0.096±0.002** | **0.092±0.001** | 0.011±0.001 |
| TiSASRec | 0.185±0.003 | 0.129±0.001 | 0.104±0.002 | 0.091±0.001 | 0.021±0.001 | 0.027±0.001 |
| +Ours | **0.216±0.005** | **0.156±0.002** | **0.133±0.002** | **0.101±0.001** | **0.093±0.001** | 0.012±0.001 |
| FEARec | 0.176±0.005 | 0.128±0.001 | 0.095±0.006 | 0.075±0.003 | 0.002±0.000 | 0.005±0.000 |
| +Ours | **0.214±0.003** | **0.158±0.003** | **0.105±0.004** | **0.077±0.002** | **0.075±0.001** | 0.003±0.000 |
| BSARec | 0.119±0.018 | 0.076±0.018 | 0.084±0.002 | 0.075±0.001 | 0.000±0.000 | 0.000±0.000 |
| +Ours | **0.219±0.003** | **0.151±0.002** | 0.067±0.003 | 0.019±0.001 | **0.079±0.001** | **0.008±0.000** |




### (Weakness 4) How does Eq. (4) avoid uniform sampling bias when optimizing \ell_{pop}(\Phi), and how does tuning \gamma affect training given that \Phi and \Psi share no gradients under stop-gradient snapshot \tilde{\Phi}?

- **W4.** Uniform negatives in $\ell_{\rm pop}$ serve to estimate the *marginal* item distribution; Cor. 1 concerns the *user-conditional* recommendation score. These are different targets. Since $\Phi$ and $\Psi$ share no gradients, Adam largely cancels constant loss scaling; we will qualify the claimed role of $\gamma$.

### (Weakness 5) During chronological test evaluation, are test interactions dynamically appended to \mathcal{H}_t^v to compute \hat{\pi}(v|t), or is \mathcal{H}_t^v frozen after the validation split?

5. During test evaluation, we only use interactions of train and validation split for computing both of $f$ and $\hat{\pi}$.


# Review by VShq

### In two of the three datasets, the best performing methods are the ones on which the proposed approach is not applied? Why is that the case? In particular, on DCRec and TPAB, would the method still give additional gains?

* We selected six general recommendation backbones to evaluate the effectiveness and architectural generality of our framework, while DCRec and TPAB serve as dedicated popularity-debiasing baselines. Unlike standard backbones, these methods incorporate their own debiasing mechanisms, and integrating our sampling strategy may require adjustments to their training objectives. Although such combinations could provide additional gains, their effectiveness is not guaranteed by our theoretical results and remains to be empirically verified.

### Please provide some basic statistics of the three datasets used. While I have seen ML1M being used in some experimentation in a few papers, the popular one is ML20M. I want to see the results on this dataset to be fully convinced.

* The dataset statistics are provided in Table 4 of the Supplementary. We selected three datasets with different characteristics: sparse short-video interactions (Micro-Video), dense movie interactions (MovieLens-1M), and intermediate-density interactions (KuaiRand). This diversity allows us to evaluate the framework under different sparsity levels and temporal popularity patterns. We agree that evaluation on MovieLens-20M would further strengthen the empirical evidence and acknowledge this as a limitation.


### In the ablation studies, the hawkes popularity wins only in 8 out of 18 cases. The fact that Hawkes popularity doesn't win over static and sliding window approaches is a concern.



### The ablation is also a bit incomplete. As the entire approach of the paper is not just negative sampling based on the Hawke's process estimators only but also involve a few changes. This makes the ablation study incomplete.

### In the ablations, what would happen if you changed static popularity to be parameterized by alpha and consider a range of values 1/(popularity)^alpha for alpha and pick the best result through validation? Would the proposed approach still do better?

### Figure 3 is not referenced in the section "Performance on Head, Mid-Tail, and Tail Items". Also, in Figure 3, what is the y-axis?

### Why are the error bars an order of magnitude higher for the baselines but not for ours? This raises some concerns. This is a recurring observation in many general recommender and ours section for several of the baselines.

### Similarly, while the wins are highlighted, in some instances like BSARec, the proposed approache massively drops the resuls, why is that happening? No explanation provided.

### There are also no statistical significance of the results making it hard to know whether these improvements are real.

### It is quite disappointing that the authors do not cite any papers on Hawkes -- neither on the Hawkes process nor on other papers in recommendations using Hawkes. This is a very serious omission.


# Review by CW75

### In Table 1, BSARec+Ours drops from 0.0771 to 0.0335 Recall@10 on MovieLens-1M. Why does the framework degrade this specific combination, and can it be avoided?

### Table 3 shows sliding-window popularity outperforms Hawkes on four of six backbones on KuaiRand. What determines which estimator wins, and what is the guidance for practitioners?

### Since the final score re-injects η·log π̂, how much of the gain comes from debiased training (sampling change) versus inference-time calibration? Please clarify using the η=0 setting.


# Review by azJs

### What additional assumptions are needed to interpret the recovered density ratio as genuine user preference, rather than just a density ratio relative to population popularity? Does the learned score still reflect user-specific exposure, policy effects, or position bias?

* As you correctly and insightfully point out, we define $f^*$ not as intrinsic user preference, but as the relative likelihood of an item being selected by a given user compared with its population-level popularity. Since observed interactions are influenced by multiple factors beyond popularity, the current formulation and data alone cannot identify intrinsic preference.
* Nevertheless, by substituting Eq. (1) into the definition of $p_0(v\mid x_u(t))$, we can express the interaction distribution as
$$
p_0(v\mid x_u(t))=
\frac{\exp\{\log\pi_0(v\mid t)+f^*(x_u(t),v)\}}
{\sum_{v'}\exp\{\log\pi_0(v'\mid t)+f^*(x_u(t),v')\}}.
$$
* This decomposition is deliberately designed to accommodate extensions that explicitly model and adjust for additional factors through suitable parameterizations of $\pi_0$ and $f^*$, provided the necessary data and identification assumptions are available. We sincerely appreciate this insightful observation and will clarify the interpretation and extensibility of our framework.

### How does the framework perform when ranking is based solely on the debiased score? Since the reported results use the calibrated score with popularity added back, it would help to know which gains are actually attributable to the debiased component alone.

### Can the authors explain why BSARec degrades so substantially on MovieLens-1M after applying the framework? This case would help clarify the practical robustness of the framework across different backbones.

### Given that sliding-window popularity sometimes beats the Hawkes estimator, under what conditions do the authors expect Hawkes to be preferable? A short characterization in terms of burstiness or temporal dynamics would help position the method more precisely.

### Could the authors clarify the intended logical role of Theorem 1? As I read it, the theorem shows how existing embedding concentration propagates to items, but does not prove that popularity confounding is the cause of that concentration in the first place.