"""
Tests for the ICL trainers with different models.
"""

import pytest
import torch
from unittest.mock import MagicMock

from src.training.icl_trainer import ICLDataCollator, MooreICLTrainer, TrainingConfig
from src.training.vanilla_rnn_trainer import MooreVanillaRNNTrainer, ICLDataCollator as VanillaRNNCollator, TrainingConfig as VanillaRNNTrainingConfig
from src.training.lstm_trainer import MooreLSTMTrainer, ICLDataCollator as LSTMCollator, TrainingConfig as LSTMTrainingConfig
from src.models.moore_vanilla_rnn import create_moore_vanilla_rnn
from src.models.moore_lstm import create_moore_lstm
from src.models.moore_transformer import MooreTransformer, TransformerConfig


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 20, (50,)),
            "target_ids": torch.randint(0, 5, (50,)),
            "loss_mask": torch.randint(0, 2, (50,)).bool(),
        }


@pytest.fixture
def mock_datasets():
    """Create mock train and validation datasets."""
    return MockDataset(100), MockDataset(50)


@pytest.fixture
def collator():
    """Create data collator."""
    return ICLDataCollator(pad_token_id=0)


@pytest.fixture
def training_config():
    """Create training configuration."""
    return VanillaRNNTrainingConfig(
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=1,  # Short for testing
        device="cpu",
        verbose=False,  # Disable verbose for testing
    )


class TestICLDataCollator:
    """Test the ICL data collator."""
    
    def test_collator(self, collator):
        """Test data collation."""
        # Create sample batch with different sequence lengths
        batch = [
            {
                "input_ids": torch.randint(0, 20, (30,)),
                "target_ids": torch.randint(0, 5, (30,)),
                "loss_mask": torch.randint(0, 2, (30,)).bool(),
            },
            {
                "input_ids": torch.randint(0, 20, (45,)),
                "target_ids": torch.randint(0, 5, (45,)),
                "loss_mask": torch.randint(0, 2, (45,)).bool(),
            },
        ]
        
        result = collator(batch)
        
        # Check output format
        assert "input_ids" in result
        assert "target_ids" in result
        assert "loss_mask" in result
        assert "attention_mask" in result
        
        # Check shapes - should be padded to max length (45)
        assert result["input_ids"].shape == (2, 45)
        assert result["target_ids"].shape == (2, 45)
        assert result["loss_mask"].shape == (2, 45)
        assert result["attention_mask"].shape == (2, 45)
        
        # Check padding
        assert result["input_ids"][0, 30:].eq(0).all()  # First sequence padded after 30
        assert result["attention_mask"][0, 30:].eq(0).all()  # Attention mask false for padding


class TestMooreICLTrainer:
    """Test the ICL trainer with different models."""
    
    def test_trainer_with_vanilla_rnn(self, mock_datasets, training_config):
        """Test trainer with Vanilla RNN."""
        train_dataset, val_dataset = mock_datasets
        collator = VanillaRNNCollator(pad_token_id=0)
        
        model = create_moore_vanilla_rnn(
            vocab_size=20,
            num_states=5,
            d_model=32,  # Smaller for faster testing
            num_layers=1,
        )
        
        trainer = MooreVanillaRNNTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            config=training_config,
        )
        
        # Should be able to create trainer without errors
        assert trainer.model is not None
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
    
    def test_trainer_with_lstm(self, mock_datasets, training_config):
        """Test trainer with LSTM."""
        train_dataset, val_dataset = mock_datasets
        collator = LSTMCollator(pad_token_id=0)
        
        model = create_moore_lstm(
            vocab_size=20,
            num_states=5,
            d_model=32,  # Smaller for faster testing
            num_layers=1,
        )
        
        trainer = MooreLSTMTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            config=training_config,
        )
        
        # Should be able to create trainer without errors
        assert trainer.model is not None
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
    
    def test_trainer_with_transformer(self, mock_datasets, collator, training_config):
        """Test trainer with Transformer."""
        train_dataset, val_dataset = mock_datasets
        
        config = TransformerConfig(
            vocab_size=20,
            num_states=5,
            d_model=32,  # Smaller for faster testing
            num_heads=4,
            num_layers=1,
        )
        model = MooreTransformer(config)
        
        trainer = MooreICLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            config=training_config,
        )
        
        # Should be able to create trainer without errors
        assert trainer.model is not None
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
    
    def test_training_step(self, mock_datasets, training_config):
        """Test a single training step."""
        train_dataset, val_dataset = mock_datasets
        collator = VanillaRNNCollator(pad_token_id=0)
        
        model = create_moore_vanilla_rnn(
            vocab_size=20,
            num_states=5,
            d_model=16,  # Very small for fast testing
            num_layers=1,
        )
        
        trainer = MooreVanillaRNNTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            config=training_config,
        )
        
        # Test single epoch (should not crash)
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        trainer.train()
        
        # Parameters should have changed after training
        final_params = {name: param for name, param in model.named_parameters()}
        
        # At least some parameters should have changed
        params_changed = False
        for name in initial_params:
            if not torch.allclose(initial_params[name], final_params[name], atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "No parameters were updated during training"
    
    def test_evaluation(self, mock_datasets, training_config):
        """Test evaluation."""
        train_dataset, val_dataset = mock_datasets
        collator = VanillaRNNCollator(pad_token_id=0)
        
        model = create_moore_vanilla_rnn(
            vocab_size=20,
            num_states=5,
            d_model=16,
            num_layers=1,
        )
        
        trainer = MooreVanillaRNNTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            config=training_config,
        )
        
        # Test evaluation
        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0  # Loss should be non-negative


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_config_creation(self):
        """Test config creation with default values."""
        config = TrainingConfig()
        
        assert config.batch_size == 8
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0
        assert config.num_epochs == 3
        assert config.device is None
    
    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=5e-4,
            weight_decay=0.01,
            num_epochs=10,
            device="cuda",
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 5e-4
        assert config.weight_decay == 0.01
        assert config.num_epochs == 10
        assert config.device == "cuda"


if __name__ == "__main__":
    pytest.main([__file__])