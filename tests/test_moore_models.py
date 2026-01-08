"""
Tests for Moore machine models (Vanilla RNN, LSTM, Transformer).
"""

import pytest
import torch

from src.models.moore_vanilla_rnn import MooreVanillaRNN, VanillaRNNConfig, create_moore_vanilla_rnn
from src.models.moore_lstm import MooreLSTM, LSTMConfig, create_moore_lstm
from src.models.moore_transformer import MooreTransformer, TransformerConfig


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "vocab_size": 20,
        "num_states": 5,
        "max_seq_len": 100,
        "d_model": 64,
        "num_layers": 2,
        "dropout": 0.1,
    }


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    batch_size, seq_len = 4, 50
    vocab_size, num_states = 20, 5
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, num_states, (batch_size, seq_len))
    unknown_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
    
    return {
        "input_ids": input_ids,
        "targets": targets,
        "unknown_mask": unknown_mask,
    }


class TestMooreVanillaRNN:
    """Test suite for Moore Vanilla RNN model."""
    
    def test_config_creation(self, sample_config):
        """Test VanillaRNNConfig creation."""
        config = VanillaRNNConfig(**sample_config, activation="tanh")
        assert config.vocab_size == 20
        assert config.num_states == 5
        assert config.activation == "tanh"
    
    def test_model_creation(self, sample_config):
        """Test model instantiation."""
        config = VanillaRNNConfig(**sample_config)
        model = MooreVanillaRNN(config)
        
        assert isinstance(model, MooreVanillaRNN)
        assert model.config.vocab_size == 20
        assert model.config.num_states == 5
    
    def test_factory_function(self, sample_config):
        """Test factory function."""
        model = create_moore_vanilla_rnn(**sample_config)
        assert isinstance(model, MooreVanillaRNN)
    
    def test_forward_pass(self, sample_config, sample_batch):
        """Test forward pass."""
        config = VanillaRNNConfig(**sample_config)
        model = MooreVanillaRNN(config)
        
        # Test without targets
        logits, loss = model(sample_batch["input_ids"])
        assert logits.shape == (4, 50, 5)  # (batch, seq_len, num_states)
        assert loss is None
        
        # Test with targets and mask
        logits, loss = model(
            sample_batch["input_ids"],
            targets=sample_batch["targets"],
            unknown_mask=sample_batch["unknown_mask"]
        )
        assert logits.shape == (4, 50, 5)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
    
    def test_activation_functions(self, sample_config, sample_batch):
        """Test different activation functions."""
        for activation in ["tanh", "relu"]:
            config = VanillaRNNConfig(**sample_config, activation=activation)
            model = MooreVanillaRNN(config)
            
            logits, _ = model(sample_batch["input_ids"])
            assert logits.shape == (4, 50, 5)
    
    def test_invalid_activation(self, sample_config):
        """Test invalid activation function."""
        config = VanillaRNNConfig(**sample_config, activation="invalid")
        model = MooreVanillaRNN(config)
        
        with pytest.raises(ValueError, match="Unknown activation"):
            model(torch.randint(0, 20, (2, 10)))
    
    def test_sequence_length_validation(self, sample_config):
        """Test sequence length validation."""
        # Create config with short max_seq_len
        short_config = sample_config.copy()
        short_config['max_seq_len'] = 10
        config = VanillaRNNConfig(**short_config)
        model = MooreVanillaRNN(config)
        
        # Should work
        short_input = torch.randint(0, 20, (2, 5))
        model(short_input)
        
        # Should fail
        long_input = torch.randint(0, 20, (2, 15))
        with pytest.raises(AssertionError):
            model(long_input)


class TestMooreLSTM:
    """Test suite for Moore LSTM model."""
    
    def test_config_creation(self, sample_config):
        """Test LSTMConfig creation."""
        config = LSTMConfig(**sample_config, bidirectional=True)
        assert config.vocab_size == 20
        assert config.num_states == 5
        assert config.bidirectional == True
    
    def test_model_creation(self, sample_config):
        """Test model instantiation."""
        config = LSTMConfig(**sample_config)
        model = MooreLSTM(config)
        
        assert isinstance(model, MooreLSTM)
        assert model.config.vocab_size == 20
        assert model.config.num_states == 5
    
    def test_factory_function(self, sample_config):
        """Test factory function."""
        model = create_moore_lstm(**sample_config)
        assert isinstance(model, MooreLSTM)
    
    def test_forward_pass(self, sample_config, sample_batch):
        """Test forward pass."""
        config = LSTMConfig(**sample_config)
        model = MooreLSTM(config)
        
        # Test without targets
        logits, loss = model(sample_batch["input_ids"])
        assert logits.shape == (4, 50, 5)  # (batch, seq_len, num_states)
        assert loss is None
        
        # Test with targets and mask
        logits, loss = model(
            sample_batch["input_ids"],
            targets=sample_batch["targets"],
            unknown_mask=sample_batch["unknown_mask"]
        )
        assert logits.shape == (4, 50, 5)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
    
    def test_bidirectional(self, sample_config, sample_batch):
        """Test bidirectional LSTM."""
        config = LSTMConfig(**sample_config, bidirectional=True)
        model = MooreLSTM(config)
        
        logits, _ = model(sample_batch["input_ids"])
        assert logits.shape == (4, 50, 5)  # Output should still be num_states
    
    def test_sequence_length_validation(self, sample_config):
        """Test sequence length validation."""
        # Create config with short max_seq_len
        short_config = sample_config.copy()
        short_config['max_seq_len'] = 10
        config = LSTMConfig(**short_config)
        model = MooreLSTM(config)
        
        # Should work
        short_input = torch.randint(0, 20, (2, 5))
        model(short_input)
        
        # Should fail
        long_input = torch.randint(0, 20, (2, 15))
        with pytest.raises(AssertionError):
            model(long_input)


class TestMooreTransformer:
    """Test suite for Moore Transformer model."""
    
    def test_model_creation(self, sample_config):
        """Test model instantiation."""
        config = TransformerConfig(**sample_config, num_heads=8)
        model = MooreTransformer(config)
        
        assert isinstance(model, MooreTransformer)
        assert model.config.vocab_size == 20
        assert model.config.num_states == 5
    
    def test_forward_pass(self, sample_config, sample_batch):
        """Test forward pass."""
        config = TransformerConfig(**sample_config, num_heads=8)
        model = MooreTransformer(config)
        
        # Test without targets
        logits, loss = model(sample_batch["input_ids"])
        assert logits.shape == (4, 50, 5)  # (batch, seq_len, num_states)
        assert loss is None
        
        # Test with targets and mask
        logits, loss = model(
            sample_batch["input_ids"],
            targets=sample_batch["targets"],
            unknown_mask=sample_batch["unknown_mask"]
        )
        assert logits.shape == (4, 50, 5)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar


class TestModelComparison:
    """Test that all models have consistent interfaces."""
    
    def test_interface_consistency(self, sample_config, sample_batch):
        """Test that all models have the same interface."""
        # Create all models
        vanilla_rnn = create_moore_vanilla_rnn(**sample_config)
        lstm = create_moore_lstm(**sample_config)
        transformer = MooreTransformer(TransformerConfig(**sample_config, num_heads=8))
        
        models = [vanilla_rnn, lstm, transformer]
        
        for model in models:
            # Test forward pass consistency
            logits, loss = model(
                sample_batch["input_ids"],
                targets=sample_batch["targets"],
                unknown_mask=sample_batch["unknown_mask"]
            )
            
            # All models should produce the same output shape
            assert logits.shape == (4, 50, 5)
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
    
    def test_parameter_counts(self, sample_config):
        """Test that models have reasonable parameter counts."""
        vanilla_rnn = create_moore_vanilla_rnn(**sample_config)
        lstm = create_moore_lstm(**sample_config)
        transformer = MooreTransformer(TransformerConfig(**sample_config, num_heads=8))
        
        # Count parameters
        vanilla_params = sum(p.numel() for p in vanilla_rnn.parameters())
        lstm_params = sum(p.numel() for p in lstm.parameters())
        transformer_params = sum(p.numel() for p in transformer.parameters())
        
        # All should have reasonable parameter counts (> 0)
        assert vanilla_params > 0
        assert lstm_params > 0
        assert transformer_params > 0
        
        # LSTM should typically have more parameters than vanilla RNN
        # Transformer should typically have the most parameters
        print(f"Vanilla RNN: {vanilla_params:,} parameters")
        print(f"LSTM: {lstm_params:,} parameters") 
        print(f"Transformer: {transformer_params:,} parameters")


if __name__ == "__main__":
    pytest.main([__file__])