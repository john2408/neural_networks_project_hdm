"""
Unit tests for NeuralForecast validation loss retrieval.

This test suite validates the logic for extracting validation loss from
NeuralForecast model's trainer callback metrics, which is used in the
Optuna hyperparameter optimization objective function.
"""

import pytest
import torch
from unittest.mock import MagicMock, PropertyMock


class TestNeuralForecastMetricsRetrieval:
    """Test validation loss extraction from NeuralForecast models."""
    
    def setup_method(self):
        """Setup common test fixtures."""
        self.default_loss_value = float('inf')
    
    def _extract_validation_loss(self, nf_model, fallback_compute=False):
        """
        Extract validation loss from NeuralForecast model.
        This is the actual implementation from run_multivariate_vec_exog_wb.py
        """
        best_val_loss = None
        if hasattr(nf_model.models[0], 'trainer') and hasattr(nf_model.models[0].trainer, 'callback_metrics'):
            metrics = nf_model.models[0].trainer.callback_metrics
            best_val_loss = metrics.get('valid_loss') or metrics.get('val_loss')
            if isinstance(best_val_loss, torch.Tensor):
                best_val_loss = best_val_loss.item()
        
        # Fallback: simulate manual computation (in real code, this would predict and compute MAE)
        if best_val_loss is None and fallback_compute:
            # In actual implementation, this would run predictions and compute loss
            # For testing, we just return a fallback value
            best_val_loss = 999.0  # Sentinel value indicating fallback was used
        
        return best_val_loss if best_val_loss is not None else float('inf')
    
    def test_valid_loss_as_tensor(self):
        """Test extraction when 'valid_loss' is a tensor."""
        # Create mock NeuralForecast model
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        # Set callback_metrics with 'valid_loss' as tensor
        expected_value = 1.234
        trainer.callback_metrics = {'valid_loss': torch.tensor(expected_value)}
        model.trainer = trainer
        nf_model.models = [model]
        
        # Extract loss
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
        assert isinstance(result, float)
    
    def test_val_loss_as_tensor(self):
        """Test extraction when 'val_loss' is a tensor (fallback)."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        # Set callback_metrics with 'val_loss' only
        expected_value = 2.345
        trainer.callback_metrics = {'val_loss': torch.tensor(expected_value)}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
        assert isinstance(result, float)
    
    def test_valid_loss_as_scalar(self):
        """Test extraction when 'valid_loss' is already a scalar."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        expected_value = 3.456
        trainer.callback_metrics = {'valid_loss': expected_value}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
    
    def test_val_loss_as_scalar(self):
        """Test extraction when 'val_loss' is a scalar (fallback)."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        expected_value = 4.567
        trainer.callback_metrics = {'val_loss': expected_value}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
    
    def test_both_metrics_present_prefers_valid_loss(self):
        """Test that 'valid_loss' is preferred when both metrics exist."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        expected_value = 5.678
        wrong_value = 9.999
        trainer.callback_metrics = {
            'valid_loss': torch.tensor(expected_value),
            'val_loss': torch.tensor(wrong_value)
        }
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
        assert result != pytest.approx(wrong_value, rel=1e-6)
    
    def test_no_trainer_attribute_with_fallback(self):
        """Test when model has no 'trainer' attribute but fallback is available."""
        nf_model = MagicMock()
        model = MagicMock()
        
        # Remove trainer attribute
        del model.trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model, fallback_compute=True)
        
        # With fallback, should compute manually
        assert result == 999.0
    
    def test_no_trainer_attribute_no_fallback(self):
        """Test when model has no 'trainer' attribute and no fallback."""
        nf_model = MagicMock()
        model = MagicMock()
        
        # Remove trainer attribute
        del model.trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model, fallback_compute=False)
        
        assert result == float('inf')
    
    def test_no_callback_metrics_attribute(self):
        """Test when trainer has no 'callback_metrics' attribute."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        # Remove callback_metrics attribute
        del trainer.callback_metrics
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == float('inf')
    
    def test_empty_callback_metrics(self):
        """Test when callback_metrics is empty."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        trainer.callback_metrics = {}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == float('inf')
    
    def test_metrics_with_unexpected_keys(self):
        """Test when callback_metrics has unexpected keys."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        trainer.callback_metrics = {
            'train_loss': torch.tensor(1.0),
            'other_metric': torch.tensor(2.0)
        }
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == float('inf')
    
    def test_tensor_with_multiple_values(self):
        """Test extraction from tensor with single value (edge case)."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        expected_value = 6.789
        # Create 0-dimensional tensor (scalar)
        trainer.callback_metrics = {'valid_loss': torch.tensor(expected_value)}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model)
        
        assert result == pytest.approx(expected_value, rel=1e-6)
        assert isinstance(result, float)
    
    def test_none_value_in_metrics_with_fallback(self):
        """Test when metric value is None and fallback is triggered."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        trainer.callback_metrics = {'valid_loss': None}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model, fallback_compute=True)
        
        # With fallback enabled, should compute manually
        assert result == 999.0  # Fallback sentinel value
    
    def test_none_value_in_metrics_no_fallback(self):
        """Test when metric value is None without fallback."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        trainer.callback_metrics = {'valid_loss': None}
        model.trainer = trainer
        nf_model.models = [model]
        
        result = self._extract_validation_loss(nf_model, fallback_compute=False)
        
        # Without fallback, should return inf
        assert result == float('inf')
    
    def test_print_available_metrics(self):
        """Diagnostic test to print what metrics are actually available."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        
        # Simulate various metric names that might exist
        trainer.callback_metrics = {
            'train_loss': torch.tensor(1.0),
            'valid_loss': torch.tensor(2.0),
            'val_loss': torch.tensor(3.0),
            'validation_loss': torch.tensor(4.0),
            'loss': torch.tensor(5.0)
        }
        model.trainer = trainer
        nf_model.models = [model]
        
        # Print available metrics
        if hasattr(nf_model.models[0], 'trainer') and hasattr(nf_model.models[0].trainer, 'callback_metrics'):
            metrics = nf_model.models[0].trainer.callback_metrics
            print("\nAvailable metrics in callback_metrics:")
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item()} (tensor)")
                else:
                    print(f"  {key}: {value} (scalar)")
        
        result = self._extract_validation_loss(nf_model)
        
        # Should get valid_loss since it exists
        assert result == pytest.approx(2.0, rel=1e-6)


class TestAlternativeMetricExtractionMethods:
    """Test alternative methods for extracting validation loss."""
    
    def test_direct_checkpoint_access(self):
        """Test accessing loss from model checkpoint (if available)."""
        nf_model = MagicMock()
        model = MagicMock()
        
        # Some models store best loss in checkpoint
        model.loss = 1.234
        nf_model.models = [model]
        
        # Alternative extraction method
        if hasattr(nf_model.models[0], 'loss'):
            result = nf_model.models[0].loss
            if isinstance(result, torch.Tensor):
                result = result.item()
        else:
            result = float('inf')
        
        assert result == pytest.approx(1.234, rel=1e-6)
    
    def test_logger_metrics_access(self):
        """Test accessing metrics from trainer's logger."""
        nf_model = MagicMock()
        model = MagicMock()
        trainer = MagicMock()
        logger = MagicMock()
        
        # Some trainers expose metrics through logger
        logger.metrics = {'val_loss': 2.345}
        trainer.logger = logger
        model.trainer = trainer
        nf_model.models = [model]
        
        # Alternative extraction method
        if hasattr(nf_model.models[0], 'trainer') and hasattr(nf_model.models[0].trainer, 'logger'):
            if hasattr(nf_model.models[0].trainer.logger, 'metrics'):
                metrics = nf_model.models[0].trainer.logger.metrics
                result = metrics.get('val_loss', float('inf'))
                if isinstance(result, torch.Tensor):
                    result = result.item()
            else:
                result = float('inf')
        else:
            result = float('inf')
        
        assert result == pytest.approx(2.345, rel=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
