# Raw NumPy XSum Summarizer

This project is an educational, end-to-end implementation of abstractive text
summarization with a one-layer Seq2Seq LSTM and Bahdanau additive attention.  The
model, loss, backpropagation through time, optimizer, checkpointing, greedy
decoding, and ROUGE metrics are written directly in NumPy and the Python standard
library.

It does not use PyTorch, TensorFlow, Keras, Hugging Face, scikit-learn, autograd,
sentence ranking, copied article sentences, or an extractive fallback.

## Architecture

```text
XSum document IDs -> encoder embedding -> encoder LSTM states
                                           |
<SOS>/previous generated ID -> decoder embedding
                                           |
encoder states + previous decoder state -> Bahdanau attention -> context
                                           |
                    [word embedding; context] -> decoder LSTM
                                           |
                    [decoder state ; attended context] -> linear projection
                                           |
                                  softmax -> next word
```

During training, teacher forcing supplies the reference previous word.  During
inference, the decoder starts with `<SOS>` and feeds back only its own argmax token
until `<EOS>` or the maximum length is reached.  The output projection scores every
token in the vocabulary, so generation is abstractive rather than sentence
selection.

## Files

- `data_loader.py` validates and loads only `xsum_train.json`,
  `xsum_validation.json`, and `xsum_test.json`.
- `preprocessing.py` provides dependency-free Unicode cleaning and tokenization.
- `embedding.py` contains `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, vocabulary frequency
  construction/mappings, and manual embedding lookup/backward accumulation.
- `lstm.py` implements the four gates, cell/hidden equations, caches, and BPTT.
- `encoder.py` runs the source embedding and encoder LSTM.
- `attention.py` implements Bahdanau energy, softmax weights, context, and backward
  gradients for `W_h`, `W_s`, `v_a`, and `b_a`.
- `decoder.py` implements teacher-forced decoding, autoregressive steps, a
  direct attended-context output projection, and complete decoder/attention
  backward flow.
- `loss.py` implements stable log-sum-exp cross-entropy from logits.
- `optimizer.py` implements SGD and bias-corrected Adam with global-norm clipping.
- `train.py` builds the training-only vocabulary, accumulates mini-batch gradients,
  validates on the predefined validation split, and saves a checkpoint.
- `inference.py` performs neural-only greedy generation from a new article.
- `evaluation.py` implements ROUGE-1, ROUGE-2, ROUGE-L, and optional SVG plots.
- `tests.py` checks analytical gradients, XSum loading, training, checkpoints,
  inference, metrics, and the absence of forbidden framework/legacy paths.

## Data

Keep these supplied files beside the Python modules:

```text
xsum_train.json
xsum_validation.json
xsum_test.json
```

Each item must contain string fields `document`, `summary`, and `id`.  The loader
uses the official predefined split files and never creates a random split.  A few
supplied records have an empty or token-empty document/summary; normal loading
warns once and skips them.  Pass `--strict-data` to make training fail on the first
invalid record.

## Requirement

Python 3.10+ and NumPy are sufficient:

```bash
python -m pip install numpy
```

## Train

From this directory, the requested default command is:

```bash
python train.py
```

It reads `xsum_train.json` and `xsum_validation.json`, builds the vocabulary from
training text only, and writes an XSum-only checkpoint to `checkpoints/xsum_model`.
The practical defaults are eight epochs (with validation early stopping), Adam
at `0.001`, embedding/hidden/attention sizes `48/64/48`, vocabulary size
`12,000`, article length `256`, summary length `40`, and batch size `4`.
These values reflect the supplied JSON: the median article is about 299 tokens,
while the median summary is 21 tokens.  The old article cap of 120 retained only
about 30% of source tokens and truncated 85.9% of valid records.
Use `--max-article-length 400` when source coverage matters more than raw NumPy
training time (it retains roughly 74% of article tokens in this corpus).

Each completed epoch is saved.  When validation is enabled, the main directory
contains the best checkpoint and `checkpoints/xsum_model/last` contains the
latest one.  `<PAD>` and `<UNK>` reference targets are excluded from the loss;
inference already masks both tokens, so the training and decoding objectives now
agree.

Raw Python/NumPy recurrent training over the full 29,994 token-valid training
records is intentionally slow.  This small command is useful for checking the
complete pipeline first:

```bash
python train.py \
  --train-limit 8 --validation-limit 2 --epochs 2 \
  --embedding-dim 8 --hidden-size 12 --attention-size 8 \
  --vocabulary-size 300 --max-article-length 30 --max-summary-length 10 \
  --batch-size 2 --checkpoint-dir checkpoints/smoke --debug-first-sample
```

To continue the supplied one-epoch checkpoint without throwing away its learned
weights (the optimizer moments are intentionally restarted and reported as a
warm start), run:

```bash
python train.py \
  --warm-start checkpoints/xsum_model \
  --checkpoint-dir checkpoints/xsum_model_v3 \
  --epochs 7 --max-article-length 256 --max-summary-length 40
```

Then point inference at `checkpoints/xsum_model_v3` (or replace the old
checkpoint after you have inspected the new validation score).

For the best capacity, start a fresh run with the default command.  The existing
checkpoint is only one epoch old (train cross-entropy 5.87, validation 5.41), so
it will continue to produce generic text until it is warm-started or retrained.

Mini-batches contain variable-length samples.  The code performs a complete
forward/backward pass for each sample, averages the accumulated gradients, then
applies one SGD/Adam update.  This avoids pretending that zero embeddings alone
correctly mask padded recurrent states.

## Infer

After training the default checkpoint:

```bash
python inference.py
```

The command prompts for an article.  Text can also be supplied directly:

```bash
python inference.py --article "A new article to summarize goes here."
python inference.py --article-file article.txt
```

Greedy decoding remains neural-only: no source copying, canned sentence, or
fallback summary is used.  By default it masks structural/OOV tokens, applies a
1.1 repetition penalty, blocks repeated 3-grams, and requires three generated
tokens before `<EOS>`.  These constraints prevent a collapsed checkpoint from
printing an endless phrase loop; they cannot replace training.  Use
`--no-repeat-ngram-size 0 --repetition-penalty 1.0` to disable them for an
unconstrained diagnostic.  If an article is passed directly from a shell, use
`$'first line\nsecond line'` or `--article-file`; the CLI also normalizes literal
`\\n` sequences defensively.

For a checkpoint written somewhere else, add `--checkpoint-dir checkpoints/smoke`.
Inference never reads a reference summary.

## Evaluate

ROUGE is calculated between neural generations and references from
`xsum_test.json` only:

```bash
python evaluation.py --samples 10
```

Omit `--samples` to evaluate every valid test record.  To save attention for the
first evaluated record:

```bash
python evaluation.py --samples 10 --heatmap checkpoints/xsum_model/attention.svg
```

## Verify

```bash
python tests.py
```

The finite-difference checks compare selected output, decoder LSTM, attention,
encoder LSTM, and embedding gradients against numerical central differences.

## Example output

On the one-record overfit smoke check (`company announced a new solar
investment`), the trained decoder produced:

```text
Generated summary:
company invests in solar
```
