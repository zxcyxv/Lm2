# Aborted tempered-readout run

The run was stopped after its step-250 report because the learned `beta`
coefficient was introduced before establishing the base full-innovation
mechanism.  This is a scope correction, not an architecture verdict.

The preserved metrics show that by step 250 H1 validation NLL and attached
latent error had improved substantially and four-step mean state cosine was
high, while H2--H4 byte agreement remained low.  Those numbers remain scoped
to the prematurely tempered readout and must not be attributed to the
replacement full-innovation experiment.

The producer, checkpoint, TSV rows, and manifest are retained.  The
replacement experiment removes the coefficient entirely from the active
readout and recurrent path.
