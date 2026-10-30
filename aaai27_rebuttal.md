# Review by tCJC

### (Weakness 1) Why does the pure debiased score (\eta=0 in Figure 4) collapse to ~0.00 for BSARec and MF on MovieLens-1M, and how does \eta=0 perform on KuaiRand's unbiased random-exposure subset?

1. By Eq. (3), observed interactions follow $p_0=\pi_0\exp f^*$. Thus, $\eta=0$ estimates a density ratio rather than the observed interaction probability, explaining its generally lower Recall across backbones, not just BSARec/MF. Reintroducing popularity at inference is therefore theoretically motivated: $\eta=0.5$ recovers the ranking of $p_0$ under exact estimation. Consistently, Fig. 4 generally peaks near $\eta=0.3$–$0.5$, outperforming popularity-only scoring ($\eta=1$). KuaiRand's test set already uses random exposure, which does not eliminate popularity-dependent user choices. Hence, low Recall at $\eta=0$ does not imply unsuccessful debiasing.


### (Weakness 2) Why does the Hawkes estimator trail Static/Sliding-window estimators on 9 of 12 cases across MovieLens-1M and KuaiRand (Table 3) and cause BSARec to drop from 0.0771 to 0.0335 on MovieLens-1M?



### (Weakness 3) From your existing Table 1 logs, what is the exact Head and Tail Recall@10 of the Vanilla Backbones compared to Backbone+(Ours)?



| **Model** | **Micro Video** | | **MovieLens** | | **KuaiRand** | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Head** | **Tail** | **Head** | **Tail** | **Head** | **Tail** |
| SASRec | .174±.006 | .124±.002 | .106±.002 | .087±.002 | .020±.002 | .027±.001 |
| +Ours | **.196±.004** | **.147±.003** | **.132±.003** | **.096±.002** | **.092±.001** | .011±.001 |
| TiSASRec | .185±.003 | .129±.001 | .104±.002 | .091±.001 | .021±.001 | .027±.001 |
| +Ours | **.216±.005** | **.156±.002** | **.133±.002** | **.101±.001** | **.093±.001** | .012±.001 |
| FEARec | .176±.005 | .128±.001 | .095±.006 | .075±.003 | .002±.000 | .005±.000 |
| +Ours | **.214±.003** | **.158±.003** | **.105±.004** | **.077±.002** | **.075±.001** | .003±.000 |
| BSARec | .119±.018 | .076±.018 | .084±.002 | .075±.001 | .000±.000 | .000±.000 |
| +Ours | **.219±.003** | **.151±.002** | .067±.003 | .019±.001 | **.079±.001** | **.008±.000** |




### (Weakness 4) How does Eq. (4) avoid uniform sampling bias when optimizing \ell_{pop}(\Phi), and how does tuning \gamma affect training given that \Phi and \Psi share no gradients under stop-gradient snapshot \tilde{\Phi}?

* For $\ell_{\mathrm{pop}}$, the population minimizer of the sampled classification loss satisfies $\lambda_v(t)\propto\pi_0(v\mid t)$ under uniform negative sampling and correct model specification. Thus, unlike in density-ratio learning, uniform sampling does not introduce a popularity term into the estimation target, although finite-sample approximation errors remain.

* Regarding $\gamma$, the absence of shared gradients is intentional, as $\Phi$ and $\Psi$ solve distinct estimation problems. We use $\gamma$ to weight their losses within a single optimizer. For $0<\gamma<1$, this weighting does not alter their respective minimizers, and Adam largely normalizes the gradient scaling. The two components are instead coupled through periodic snapshot updates and negative resampling. Thus, cross-model gradient flow is neither required nor intended.


### (Weakness 5) During chronological test evaluation, are test interactions dynamically appended to \mathcal{H}_t^v to compute \hat{\pi}(v|t), or is \mathcal{H}_t^v frozen after the validation split?

* During test evaluation, we only use interactions of train and validation split for computing both of $f$ and $\hat{\pi}$. Further implementation details are available in the attached code.


# Review by VShq

### In two of the three datasets, the best performing methods are the ones on which the proposed approach is not applied? Why is that the case? In particular, on DCRec and TPAB, would the method still give additional gains?

* We selected six general recommendation backbones to evaluate the effectiveness and architectural generality of our framework, while DCRec and TPAB serve as dedicated popularity-debiasing baselines. Unlike standard backbones, these methods incorporate their own debiasing mechanisms, and integrating our sampling strategy may require adjustments to their training objectives. Although such combinations could provide additional gains, their effectiveness is not guaranteed by our theoretical results and remains to be empirically verified.

### Please provide some basic statistics of the three datasets used. While I have seen ML1M being used in some experimentation in a few papers, the popular one is ML20M. I want to see the results on this dataset to be fully convinced.

* The dataset statistics are provided in Table 4 of the Supplementary. We selected three datasets with different characteristics: sparse short-video interactions (Micro-Video), dense movie interactions (MovieLens-1M), and intermediate-density interactions (KuaiRand). This diversity allows us to evaluate the framework under different sparsity levels and temporal popularity patterns. We agree that evaluation on MovieLens-20M would further strengthen the empirical evidence and acknowledge this as a limitation.


### In the ablation studies, the hawkes popularity wins only in 8 out of 18 cases. The fact that Hawkes popularity doesn't win over static and sliding window approaches is a concern.



### The ablation is also a bit incomplete. As the entire approach of the paper is not just negative sampling based on the Hawke's process estimators only but also involve a few changes. This makes the ablation study incomplete.

* Thank you for your suggestion. Our ablation analyses are already presented across several sections: Table 1 compares backbones with and without our framework, Figure 4 examines inference-time calibration (including $\eta=0$, which uses only the debiased score), and Table 3 compares different popularity estimators. To make these analyses easier to review collectively, we will consolidate and summarize them in a dedicated appendix section.

### In the ablations, what would happen if you changed static popularity to be parameterized by alpha and consider a range of values 1/(popularity)^alpha for alpha and pick the best result through validation? Would the proposed approach still do better?

| **MF** | **Micro Video** | **MovieLens** | **KuaiRand** |
|:---|:---:|:---:|:---:|
|$\text{pop}^{-\alpha} ~(\alpha=)$|.063±.001 (.5)|.037±.000 (.5)|.068±.001 (.5)|
|Hawkes|.060±.001|.038±.001|.069±.001|


| **SASRec** | **Micro Video** | **MovieLens** | **KuaiRand** |
|:---|:---:|:---:|:---:|
|$\text{pop}^{-\alpha}~(\alpha=)$|.131±.004 (1)|.104±.001 (.5)|.071±.001 (.5)|
|Hawkes|.163±.003|.107±.002|.075±.000|

* Following the suggestion, we parameterized the popularity estimator as $\text{pop}^{-\alpha}$, and evaluated its performance using Recall@10 on the test set. We searched $\alpha \in \{0.5, 1, 1.5\}$ and also tuned $\eta$ on the same grid, selecting both by validation Recall@10. Hawkes remains better in 5 of 6 settings. The exception is MF on Micro-Video, where the two estimators were already tied in Table 3. Hence the gain is not explained by the strength of the popularity correction alone, but by modeling its temporal variation.


### Figure 3 is not referenced in the section "Performance on Head, Mid-Tail, and Tail Items". Also, in Figure 3, what is the y-axis?
* Fig.3 is discussed in the corresponding results section, and its caption specifies Recall@10. We will make the figure reference and y-axis label more explicit.

### Why are the error bars an order of magnitude higher for the baselines but not for ours? This raises some concerns. This is a recurring observation in many general recommender and ours section for several of the baselines.

### Similarly, while the wins are highlighted, in some instances like BSARec, the proposed approache massively drops the resuls, why is that happening? No explanation provided.

### There are also no statistical significance of the results making it hard to know whether these improvements are real.

### It is quite disappointing that the authors do not cite any papers on Hawkes -- neither on the Hawkes process nor on other papers in recommendations using Hawkes. This is a very serious omission.

* Thank you for pointing this out. We will add a citation to the original work [1].

[1] Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58(1), 83-90.


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

* We agree that Theorem 1 characterizes how user-embedding concentration propagates to aligned item embeddings, rather than establishing its cause. Its purpose is to highlight the geometric consequences of such concentration. Importantly, Table 2 empirically demonstrates that substantial user-embedding concentration occurs in models trained on real data and that our framework effectively mitigates it. Together with Corollaries 1–2, which establish how popularity confounding enters the learning target, these findings motivate our approach to preventing such undesirable representation geometry. We will clarify this logical connection.