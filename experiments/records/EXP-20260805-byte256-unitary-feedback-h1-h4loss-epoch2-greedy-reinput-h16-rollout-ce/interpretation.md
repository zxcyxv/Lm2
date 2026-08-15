# Epoch-2 greedy self-reinput H16 block comparison

Gold context resets only at each sixteen-token boundary. Inside it, H1 re-enters one greedy token, H4 re-enters four greedy tokens, and scan predicts sixteen open-loop. Values are gold-aligned rollout CE, not teacher-forced likelihood.

- feedback_h1loss_greedy_block1: 7.063616083
- feedback_h4loss_greedy_block4: 3.494413535
- parallel_scan_open_block16: 2.912845770
