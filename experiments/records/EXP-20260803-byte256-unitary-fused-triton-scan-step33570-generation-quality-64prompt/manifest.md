# Step-33570 checkpoint-swap generation audit

- Question: what H1 shallow continuation quality is present at epoch 10?
- Comparison: identical preserved 64-prompt producer; checkpoint only changes to step 33570, with step 10071 and step 16785 as matched references.
- Seed: 1337 with the existing fixed sampling seeds.
- Split: fixed validation starts; test unread.
- Success criterion: report H1 first-byte, first-four-byte, and shallow sample quality only; do not infer quality from later horizons.
- Status: completed.
