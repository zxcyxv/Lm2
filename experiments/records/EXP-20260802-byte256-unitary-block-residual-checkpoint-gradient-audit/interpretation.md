# Checkpoint gradient attribution

On the saved-generator next batch, step 100 had global gradient norm
`30.095601`. Central O was largest (`16.560498`, 30.28% of global squared
norm), but unlike the predecessor it did not monopolize the event: V
(`13.029069`, 18.74%), hidden phase (`12.329188`, 16.78%), and Q
(`11.284937`, 14.06%) were jointly large. Central components together
accounted for 81.69% of squared norm. The hidden phase was especially sharp
relative to its small parameter norm (`434.89x`) and had max absolute
gradient `11.853407`.

At step 200 the reproducible next-batch global norm was `7.196011`. Q was
largest (`4.238688`, 34.70%), followed by O (`3.701978`, 26.47%), V
(`2.657568`, 13.64%), hidden phase (`2.076293`, 8.33%), and K (`1.541457`,
4.59%). Central components still accounted for 87.75% of squared norm; the
encoder was secondary.

The disjoint squared sums reproduced each global norm with error at most
`1.26e-16`. These rows attribute the exact next batch after each checkpoint,
not the historical batches that produced the logged checkpoint updates.
