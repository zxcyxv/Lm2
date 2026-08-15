# Step-3000 comparison interpretation

The step-3000 selective model exceeds the preserved step-1000 GPT-2
checkpoint on teacher-forced h1 prediction under both target-free selectors:

- GPT-2 AR: NLL 4.934292, accuracy 0.216003
- selective confidence: NLL 4.578397, accuracy 0.266846
- selective prior: NLL 4.591133, accuracy 0.252777

The label-informed selective oracle remains substantially better at NLL
4.044334 and accuracy 0.301666. The candidate set therefore contains useful
headroom which the learned prior does not recover. Prior winner accuracy is
0.342773, close to random chance 1/3, and prior h1 accuracy is approximately
the candidate path mean. The improved prior-policy h1 result is chiefly an
improvement shared by all candidates rather than successful candidate
selection.

In 64-token generation, selective prior block-4 has higher distinct-2 and
lower repeated 4-gram coverage than GPT-2 AR, but it does not produce
consistently coherent prose. It exhibits more immediate and period-4
repetition than GPT-2 and degrades relative to the same selective
architecture at step 1000:

| metric | GPT-2 step 1000 | selective step 1000 | selective step 3000 |
|---|---:|---:|---:|
| distinct-2 | 0.276786 | 0.498760 | 0.482887 |
| immediate repeat | 0.093006 | 0.171379 | 0.278026 |
| period-4 repeat | 0.255729 | 0.252865 | 0.330208 |
| repeated 4-gram coverage | 0.739498 | 0.225154 | 0.336322 |
| collapsed | 0.296875 | 0.046875 | 0.234375 |

Over the same interval, innovation scale falls from 0.057641 to 0.040465
while prior winner accuracy remains near chance. The evidence supports a
separation between one-step/token prediction improvement and successful
open-loop trajectory commitment. It does not support claiming that the
step-3000 block generator has surpassed GPT-2 in overall sentence quality.
