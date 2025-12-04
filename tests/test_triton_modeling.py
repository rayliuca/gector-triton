"""
Unit tests for GECToRTriton class.

Note: These tests require a running Triton Inference Server with a deployed model.
They are provided as documentation of expected behavior and for manual testing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestGECToRTritonImport(unittest.TestCase):
    """Test that GECToRTriton can be imported when tritonclient is available."""
    
    def test_import_without_tritonclient(self):
        """Test that module handles missing tritonclient gracefully."""
        # When tritonclient is not installed, GECToRTriton should be None in __init__
        import sys
        
        # Temporarily hide tritonclient if it exists
        tritonclient_modules = {k: v for k, v in sys.modules.items() if 'tritonclient' in k}
        for module in tritonclient_modules:
            sys.modules.pop(module, None)
        
        # Force reimport
        import importlib
        import gector
        importlib.reload(gector)
        
        # GECToRTriton should either be None or not in __all__
        if hasattr(gector, 'GECToRTriton'):
            # If the import succeeded (tritonclient was available), skip this test
            self.skipTest("tritonclient is installed")
        else:
            self.assertNotIn('GECToRTriton', gector.__all__)
        
        # Restore modules
        sys.modules.update(tritonclient_modules)


class TestGECToRTritonMocked(unittest.TestCase):
    """
    Test GECToRTriton with mocked Triton client.
    
    These tests mock the Triton client to test the logic without requiring
    a running server.
    """
    
    @patch('gector.triton_modeling.TRITON_AVAILABLE', True)
    @patch('gector.triton_modeling.grpcclient')
    def test_initialization(self, mock_grpcclient):
        """Test that GECToRTriton initializes correctly."""
        from gector.triton_modeling import GECToRTriton
        from gector import GECToRConfig
        
        # Mock the Triton client
        mock_client = MagicMock()
        mock_client.is_server_live.return_value = True
        mock_client.is_model_ready.return_value = True
        mock_grpcclient.InferenceServerClient.return_value = mock_client
        
        # Create config
        config = GECToRConfig(
            label2id={'$KEEP': 0, '$DELETE': 1, '<PAD>': 2, '<OOV>': 3},
            id2label={0: '$KEEP', 1: '$DELETE', 2: '<PAD>', 3: '<OOV>'},
        )
        
        # Initialize model
        model = GECToRTriton(
            config=config,
            triton_url='localhost:8001',
            model_name='test_model',
            verbose=False
        )
        
        # Verify initialization
        self.assertEqual(model.triton_url, 'localhost:8001')
        self.assertEqual(model.model_name, 'test_model')
        self.assertIsNotNone(model.triton_client)
        
        # Verify client was created with correct URL
        mock_grpcclient.InferenceServerClient.assert_called_once_with(
            url='localhost:8001',
            verbose=False
        )
    
    @patch('gector.triton_modeling.TRITON_AVAILABLE', True)
    @patch('gector.triton_modeling.grpcclient')
    def test_forward_inference(self, mock_grpcclient):
        """Test that forward() makes correct Triton inference calls."""
        from gector.triton_modeling import GECToRTriton
        from gector import GECToRConfig
        
        # Mock the Triton client
        mock_client = MagicMock()
        mock_client.is_server_live.return_value = True
        mock_client.is_model_ready.return_value = True
        
        # Mock inference response
        mock_response = MagicMock()
        mock_response.as_numpy.side_effect = [
            np.random.randn(2, 10, 4),  # logits_labels
            np.random.randn(2, 10, 2),  # logits_d
        ]
        mock_client.infer.return_value = mock_response
        mock_grpcclient.InferenceServerClient.return_value = mock_client
        
        # Create config
        config = GECToRConfig(
            label2id={'$KEEP': 0, '$DELETE': 1, '<PAD>': 2, '<OOV>': 3},
            id2label={0: '$KEEP', 1: '$DELETE', 2: '<PAD>', 3: '<OOV>'},
        )
        
        # Initialize model
        model = GECToRTriton(
            config=config,
            triton_url='localhost:8001',
            model_name='test_model'
        )
        
        # Create sample inputs
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        
        # Call forward
        output = model.forward(input_ids, attention_mask)
        
        # Verify inference was called
        mock_client.infer.assert_called_once()
        
        # Verify output structure
        self.assertIsNotNone(output.logits_labels)
        self.assertIsNotNone(output.logits_d)
        self.assertEqual(output.logits_labels.shape, (2, 10, 4))
        self.assertEqual(output.logits_d.shape, (2, 10, 2))
    
    @patch('gector.triton_modeling.TRITON_AVAILABLE', True)
    @patch('gector.triton_modeling.grpcclient')
    def test_training_not_supported(self, mock_grpcclient):
        """Test that training mode raises NotImplementedError."""
        from gector.triton_modeling import GECToRTriton
        from gector import GECToRConfig
        
        # Mock the Triton client
        mock_client = MagicMock()
        mock_client.is_server_live.return_value = True
        mock_client.is_model_ready.return_value = True
        mock_grpcclient.InferenceServerClient.return_value = mock_client
        
        # Create config
        config = GECToRConfig(
            label2id={'$KEEP': 0, '$DELETE': 1, '<PAD>': 2, '<OOV>': 3},
            id2label={0: '$KEEP', 1: '$DELETE', 2: '<PAD>', 3: '<OOV>'},
        )
        
        # Initialize model
        model = GECToRTriton(config=config)
        
        # Create sample inputs with labels
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        labels = torch.randint(0, 4, (2, 10))
        
        # Should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            model.forward(input_ids, attention_mask, labels=labels)


class TestGECToRTritonIntegration(unittest.TestCase):
    """
    Integration tests that require a running Triton server.
    
    These tests are skipped by default and can be enabled by setting
    the TRITON_SERVER_URL environment variable.
    """
    
    def setUp(self):
        import os
        self.triton_url = os.environ.get('TRITON_SERVER_URL')
        if not self.triton_url:
            self.skipTest("TRITON_SERVER_URL not set - skipping integration tests")
        
        try:
            from gector import GECToRTriton
            self.GECToRTriton = GECToRTriton
        except ImportError:
            self.skipTest("tritonclient not installed")
    
    def test_real_server_connection(self):
        """Test connection to a real Triton server."""
        from gector import GECToRConfig
        
        # This test requires a real server to be running
        config = GECToRConfig(
            label2id={'$KEEP': 0, '$DELETE': 1, '<PAD>': 2, '<OOV>': 3},
            id2label={0: '$KEEP', 1: '$DELETE', 2: '<PAD>', 3: '<OOV>'},
        )
        
        # This will raise an exception if the server is not available
        model = self.GECToRTriton(
            config=config,
            triton_url=self.triton_url,
            model_name='gector',
            verbose=True
        )
        
        self.assertIsNotNone(model.triton_client)


if __name__ == '__main__':
    unittest.main()
