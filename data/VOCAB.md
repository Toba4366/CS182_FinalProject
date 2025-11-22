## Vocabulary Layout

Each dataset sample uses a shared token space:

- **State tokens**: `0 ... num_states-1`
- **Action tokens**: `num_states ... num_states + max_actions - 1`
- **Special tokens**:
  - `<eos>` (`eos_token`) separates demo/query segments.
  - `<query>` (`query_token`) marks the start of the query portion.
  - `<pad>` (`pad_token`) is used for batching.

During training we feed states *and* actions into the transformer, but the loss mask
only scores the query-action tokens, so the model learns to predict actions given the
preceding state history.

