# Step-16785 checkpoint swap generation audit

- Question: did the already measured H1 shallow quality change from epoch 3 to epoch 5?
- Comparison: identical existing 64-prompt producer; checkpoint only changes from step 10071 to step 16785.
- Seed: 1337 with the existing fixed sampling seeds.
- Split: fixed validation starts; test unread.
- Success criterion: report only matched H1 shallow metrics and samples; do not infer quality from later horizons.
- Status: completed.
