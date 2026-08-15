# Step-5000 byte AR versus block-4 generation

The four-byte path is not AR-equivalent at this checkpoint: mean byte
agreement is `9.90%` and the mean identical prefix is one byte. Its output is
strict-valid printable ASCII, so byte encoding itself did not break. Instead,
generation is linguistically degenerate through repeated character runs:
immediate repetition is `66.14%`, versus `1.06%` for greedy AR.

- AR aggregate: `{'distinct_1': 0.2708333333333333, 'distinct_2': 0.5502645502645502, 'immediate_repeat': 0.010582010582010581, 'ascii_printable_fraction': 1.0, 'strict_utf8_valid': 1.0, 'replacement_characters': 0.0}`
- Block-4 aggregate: `{'distinct_1': 0.19791666666666666, 'distinct_2': 0.48677248677248675, 'immediate_repeat': 0.6613756613756614, 'ascii_printable_fraction': 1.0, 'strict_utf8_valid': 1.0, 'replacement_characters': 0.0}`
- Mean per-sample block-4/AR agreement: `{'first_token_accuracy': 1.0, 'reference_accuracy': 0.09895833333333333, 'matching_prefix_tokens': 1.0, 'distinct_1': 0.19791666666666666, 'distinct_2': 0.48677248677248675, 'distinct_4': 0.907103825136612, 'immediate_repeat': 0.6613756613756614, 'period_2_repeat': 0.39784946236559143, 'period_4_repeat': 0.011111111111111112, 'repeated_bigram_coverage': 0.6825396825396824, 'repeated_4gram_coverage': 0.16393442622950818, 'longest_identical_run': 4.666666666666667, 'collapsed': 0.0}`

## Sample 1

**Prompt**

```text
ung from BBC Music called the performance a " delight " but described the video as the " lowlight " of the tour . Ed Gonzalez from Slant Magazine wrote that the performance was not among the concert 's highlights . However , he commented that the remix bac
```

**Reference**

```text
kdrop " feels ecstatic [ ... ] a collage of Godardian weight you
```

**Greedy AR**

```text
k to the song " . 

 = = = = Second World War = = = = 

 The ser
```

**Four-byte composition**

```text
ktuuppppoooo   neeeeddddlllliyyy    ttttiiiinnnnggeerrrrriiinnnn
```

## Sample 2

**Prompt**

```text
ockey Club , James B. Brown , had lost his fortune and influence when his banking empire crashed in 1930 . In 1933 Beckham 's ally , Robert Bingham , had been appointed ambassador to the Court of St. James 's , increasing his prominence and influence . Bec
```

**Reference**

```text
kham also enjoyed the support of the United Mine Workers and Lou
```

**Greedy AR**

```text
ause of the series , the state of the series , and the second se
```

**Four-byte composition**

```text
aaaauuuunnnncdddeeeensdd    @@@@----    ttttiiiinnna     aaannnn
```

## Sample 3

**Prompt**

```text
elease Harry Potter and the Prisoner of Azkaban was the third film in the series . Radcliffe 's performance was panned by New York Times journalist A. O. Scott , who wrote that Watson had to carry him with her performance . Next was Harry Potter and the Go
```

**Reference**

```text
blet of Fire in 2005 . The film was the second @-@ highest gross
```

**Greedy AR**

```text
vernor of the Constitution , and the series of the series , and 
```

**Four-byte composition**

```text
vdddi   ttttioiinnnneeee  dduuuunnrreeee dd ttttoooommkke   tttt
```
