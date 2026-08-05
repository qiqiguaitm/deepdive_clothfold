# pi0.5 R1 Dense-Target Coverage Audit

Recorded before any R1 closed-loop result was available.

## Finding

The initial R1 launch used `probe_train.npz`, which is a fixed-size offline
readout sample rather than a dense training-label artifact. It contains 8,190
exact `(episode, frame)` rows over a 6,075,103-frame policy dataset. Because
the lookup masks every unmatched row, only 0.1348% of sampled frames were
eligible for recurrence supervision. With batch size 16, the expected number
of labeled rows was 0.0216 per batch and the probability of any labeled row
was 2.14%. Aggregated training logs consequently contained 100-step windows
with exactly zero recurrence loss.

No policy evaluation or closed-loop outcome had been observed when this issue
was identified. The initial R1 runs are invalid as tests of the recurrence
objective and must not enter a scientific comparison.

## Correction

`reference_trajectories.npz` already contains frozen per-frame CRAVE fields for
all 1,200 reference episodes. The deterministic builder
`build_pi05_r1_dense_targets.py` converts every valid `t -> t+50` row into the
same three preregistered targets without changing policy-data sampling:

- 359,823 target rows over 1,200 episodes and six tasks;
- 50-frame horizon;
- 85.71% coverage within the frozen reference trajectories;
- 5.923% coverage over the unchanged full policy dataset;
- 0.9477 expected labeled rows per batch and 62.35% probability that a batch
  contains recurrence supervision.

The artifact SHA-256 is
`d795596174d5280f61e09274cd57c3930ba604fa77b130ea714ee97efbc23119`.
The regenerated R1 protocol SHA-256 is
`f9a651df824faaa14caedda7b3517fcda18e0b29fb27c611d6af6e801c29e67d`.
The protocol verifier checks both the artifact and its manifest, requires at
least 300,000 rows, and preserves the original six-task, seed, initialization,
normalization, optimizer, update-count, and closed-loop evaluation contracts.
