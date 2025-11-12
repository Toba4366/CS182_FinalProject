1. RNNs
2. LSTM
3. S4 
4. Mamba
5. Traditional Attention with RoPE (Check different number of transformer layers)

6. Linear Attention? (check back later)
7. Hybrid Models? (check back later)


Implement nn.Module (forward, __init__ (define all variables/parameters))
-> models
    -> state-space
        -> utils (for s4, mamba, hippo - initialization)
        -> rnn.py
        ...
    -> transformers (flash attention - mha)
        -> utils (rope)
        -> traditional.py
        ...
-> cmd 
    -> train.py (take in module, wandb, saving models, generate graphs)
-> fsm 
    -> class fsm (dictionary key: state value: dict{action: state})
        -> class method - random fsm
    -> solver (input: fsm, action)


Decide how many samples, what sequence lengths 
Create 10,000 samples 
Create dataset with 5000 samples for training (split in 4000 to 1000 for train:validation)
Create another test dataset with 1000 samples



256
Sequence of length 64 (A, S) pairs 