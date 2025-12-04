"""
GECToR model class that uses Triton Inference Server for remote inference.

This module provides GECToRTriton, which extends GECToR to support
inference on models served via NVIDIA Triton Inference Server.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from .modeling import GECToR, GECToROutput, GECToRPredictionOutput
from .configuration import GECToRConfig

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class GECToRTriton(GECToR):
    """
    GECToR model that uses Triton Inference Server for remote inference.
    
    This class extends the base GECToR class to support models served on a remote
    Triton Inference Server. Instead of loading the BERT/RoBERTa model locally,
    it sends inference requests to the Triton server.
    
    Args:
        config (GECToRConfig): Configuration for the GECToR model
        triton_url (str): URL of the Triton Inference Server (e.g., "localhost:8001")
        model_name (str): Name of the model on the Triton server
        model_version (str): Version of the model to use (default: "1")
        verbose (bool): Enable verbose logging for Triton client (default: False)
    
    Example:
        >>> config = GECToRConfig.from_pretrained("gotutiyan/gector-roberta-base-5k")
        >>> model = GECToRTriton(
        ...     config=config,
        ...     triton_url="localhost:8001",
        ...     model_name="gector_bert"
        ... )
        >>> # Use model for predictions as usual
    """
    
    def __init__(
        self,
        config: GECToRConfig,
        triton_url: str = "localhost:8001",
        model_name: str = "gector",
        model_version: str = "1",
        verbose: bool = False
    ):
        if not TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient is required for GECToRTriton. "
                "Install it with: pip install tritonclient[grpc]"
            )
        
        # IMPORTANT: We bypass GECToR.__init__() to avoid loading the BERT model locally.
        # Instead, we call PreTrainedModel.__init__() directly to set up the base model infrastructure.
        # This is intentional: for Triton inference, we don't need local model weights.
        super(GECToR, self).__init__(config)  # Calls PreTrainedModel.__init__
        
        self.config = config
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        
        # Initialize Triton client
        self._init_triton_client()
        
        # Note: We don't load bert, label_proj_layer, d_proj_layer, or dropout
        # since inference happens on the Triton server
    
    def _init_triton_client(self):
        """Initialize the Triton gRPC client."""
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.triton_url,
                verbose=self.verbose
            )
            
            # Check if server is live
            if not self.triton_client.is_server_live():
                raise RuntimeError(
                    f"Triton server at {self.triton_url} is not live"
                )
            
            # Check if model is ready
            if not self.triton_client.is_model_ready(
                self.model_name,
                self.model_version
            ):
                raise RuntimeError(
                    f"Model {self.model_name} (version {self.model_version}) "
                    f"is not ready on Triton server"
                )
            
            if self.verbose:
                print(f"Connected to Triton server at {self.triton_url}")
                print(f"Model {self.model_name} (version {self.model_version}) is ready")
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Triton client: {str(e)}"
            )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        d_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        word_masks: Optional[torch.Tensor] = None,
    ) -> GECToROutput:
        """
        Forward pass using Triton Inference Server.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs (not used in Triton inference)
            inputs_embeds: Input embeddings (not used in Triton inference)
            labels: Label IDs for training (not supported in Triton mode)
            d_labels: Detection labels for training (not supported in Triton mode)
            output_attentions: Not used in Triton mode
            output_hidden_states: Not used in Triton mode
            return_dict: Not used in Triton mode
            word_masks: Word masks for accuracy calculation
        
        Returns:
            GECToROutput with logits from Triton server
        """
        if labels is not None or d_labels is not None:
            raise NotImplementedError(
                "Training mode (with labels) is not supported for Triton inference"
            )
        
        # Prepare inputs for Triton
        # Convert torch tensors to numpy arrays
        input_ids_np = input_ids.cpu().numpy().astype(np.int64)
        attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)
        
        # Create input tensors for Triton
        input_ids_input = grpcclient.InferInput("input_ids", input_ids_np.shape, "INT64")
        input_ids_input.set_data_from_numpy(input_ids_np)
        
        attention_mask_input = grpcclient.InferInput("attention_mask", attention_mask_np.shape, "INT64")
        attention_mask_input.set_data_from_numpy(attention_mask_np)
        
        inputs = [input_ids_input, attention_mask_input]
        
        # Define outputs we expect from Triton
        outputs = [
            grpcclient.InferRequestedOutput("logits_labels"),
            grpcclient.InferRequestedOutput("logits_d")
        ]
        
        # Call Triton server
        try:
            response = self.triton_client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get output tensors and convert back to torch
            logits_labels_np = response.as_numpy("logits_labels")
            logits_d_np = response.as_numpy("logits_d")
            
            # Convert to torch tensors
            device = input_ids.device
            logits_labels = torch.from_numpy(logits_labels_np).to(device)
            logits_d = torch.from_numpy(logits_d_np).to(device)
            
            return GECToROutput(
                loss=None,
                loss_d=None,
                loss_labels=None,
                logits_d=logits_d,
                logits_labels=logits_labels,
                accuracy=None,
                accuracy_d=None
            )
            
        except InferenceServerException as e:
            raise RuntimeError(
                f"Triton inference failed: {str(e)}"
            )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_masks: torch.Tensor,
        keep_confidence: float = 0,
        min_error_prob: float = 0
    ) -> GECToRPredictionOutput:
        """
        Prediction method using Triton Inference Server.
        
        This method performs the same operations as the base GECToR.predict()
        but uses the Triton server for inference.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            word_masks: Word masks
            keep_confidence: Bias for the $KEEP label
            min_error_prob: Minimum error probability threshold
        
        Returns:
            GECToRPredictionOutput with predictions
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask
            )
            # (batch, seq_len, num_labels)
            probability_labels = F.softmax(outputs.logits_labels, dim=-1)
            # (batch, seq_len, num_labels)
            probability_d = F.softmax(outputs.logits_d, dim=-1)

            # Apply the bias of $KEEP.
            keep_index = self.config.label2id[self.config.keep_label]
            probability_labels[:, :, keep_index] += keep_confidence
            # Get prediction tags. (batch, seq_len, num_labels) -> (batch, seq_len)
            pred_label_ids = torch.argmax(probability_labels, dim=-1)

            # Apply the minimum error probability threshold
            incor_idx = self.config.d_label2id[self.config.incorrect_label]
            # (batch_size, seq_len, num_labels) -> (batch_size, seq_len)
            probability_d_incor = probability_d[:, :, incor_idx]
            # (batch_size, seq_len) -> (batch_size)
            max_error_probability = torch.max(probability_d_incor * word_masks, dim=-1)[0]
            # Sentence-level threshold.
            #   Set the $KEEP tag to all tokens in the sentences 
            #   that have lower maximum error prob. than threshold.
            pred_label_ids[
                max_error_probability < min_error_prob, :
            ] = keep_index
            # Token-level threshold.
            #   Set infinity to tokens that have lower probability than threshold.
            #   Note that the probability is not detection's, but tag's one.
            pred_label_ids[
                torch.max(probability_labels, dim=-1)[0] < min_error_prob
            ] = keep_index

            # Note: This helper function is duplicated from GECToR.predict()
            # to maintain consistency with the parent class implementation.
            def convert_ids_to_labels(ids, id2label):
                labels = []
                for id in ids.tolist():
                    labels.append(id2label[id])
                return labels

            pred_labels = []
            for ids in pred_label_ids:
                labels = convert_ids_to_labels(
                    ids,
                    self.config.id2label
                )
                pred_labels.append(labels)
        
        return GECToRPredictionOutput(
            probability_labels=probability_labels,
            probability_d=probability_d,
            pred_labels=pred_labels,
            pred_label_ids=pred_label_ids,
            max_error_probability=max_error_probability
        )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        triton_url: str = "localhost:8001",
        model_name: str = "gector",
        model_version: str = "1",
        verbose: bool = False,
        **kwargs
    ) -> "GECToRTriton":
        """
        Load configuration from a pretrained model and create GECToRTriton instance.
        
        Args:
            pretrained_model_name_or_path: Path or model ID to load config from
            triton_url: URL of the Triton Inference Server
            model_name: Name of the model on Triton server
            model_version: Version of the model
            verbose: Enable verbose logging
            **kwargs: Additional arguments
        
        Returns:
            GECToRTriton instance configured for Triton inference
        """
        # Load the config from the pretrained model
        config = GECToRConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create and return the Triton model
        return cls(
            config=config,
            triton_url=triton_url,
            model_name=model_name,
            model_version=model_version,
            verbose=verbose
        )
    
    def tune_bert(self, tune=True):
        """
        Override tune_bert since we don't have a local BERT model.
        This is a no-op for Triton models.
        """
        if self.verbose:
            print("tune_bert() is not applicable for Triton models - skipping")
        return
    
    def init_weight(self):
        """
        Override init_weight since we don't have local layers.
        This is a no-op for Triton models.
        """
        if self.verbose:
            print("init_weight() is not applicable for Triton models - skipping")
        return
