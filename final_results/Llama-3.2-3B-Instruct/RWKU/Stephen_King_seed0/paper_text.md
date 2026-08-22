# Paper-ready wording

## Results paragraph

On the Stephen King post-hoc RWKU development target, the untouched base vocabulary readout recovered the sensitive answer on all 48 generated atomic views (100% recovery; minimum answer demotion margin -18.8125). Our answer-level representation repair with 1K Wikipedia KL preservation reduced frozen-base-head recovery to 0% while maintaining 100% direct and generated-subject atomic success. The selected rank-1 candidate achieved a minimum frozen-readout demotion margin of +0.6961. Importantly, utility preservation generalized beyond the optimization reservoir: on a disjoint held-out set of 1,000 Wikipedia contexts, exact full-vocabulary KL was 0.000386 mean, 0.001657 p95, and 0.036320 max, all far below the predeclared limits of 0.01, 0.05, and 0.5. The candidate nevertheless exceeded the predeclared 1% intervention-size constraint, with a 1.4117% relative Frobenius change in the final-layer MLP `down_proj`, and is therefore not counted as fully feasible under the strict protocol.

## Short table caption

Stephen King post-hoc RWKU development result on Llama-3.2-3B-Instruct. v3.2 reaches 0% recovery under the untouched base readout and strongly preserves a disjoint 1K Wikipedia utility set, but exceeds the predeclared 1% final-layer representation-intervention budget.

## Discussion wording

The progression from v3.1 to v3.2 isolates the utility-preservation bottleneck. v3.1 already demonstrated that 0% frozen-base-head recovery was attainable with a sub-1% representation update, but exact Wikipedia KL remained large. Adding direct full-vocabulary KL preservation on 1,000 external-Wikipedia contexts reduced held-out KL by orders of magnitude while retaining complete generated-view suppression. Thus, in this development setting, forgetting and broad utility preservation are not intrinsically incompatible. The remaining constraint is the size of the physical representation intervention required by the selected v3.2 checkpoint.

## Limitation wording

This result should not be interpreted as proof of irreversible latent erasure. The evaluation establishes operational unrecoverability under the tested generated atomic views and an untouched base vocabulary readout. Moreover, the Stephen King target is a post-hoc development setting because official RWKU metrics had been observed in earlier experiments before v3.2 was designed. Finally, the held-out 1K Wikipedia utility set has now been opened and must not be used for further tuning; future confirmatory experiments require a fresh untouched utility set and additional targets.

## One-sentence claim

v3.2 achieves complete generated-view suppression under the original frozen decoder with strong disjoint held-out utility preservation, while missing only the strict 1% representation-intervention budget.
