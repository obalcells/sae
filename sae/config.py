from dataclasses import dataclass
from simple_parsing import list_field, Serializable
from typing import Dict, Any


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    k: int = 32
    """Number of nonzero features."""

    signed: bool = False


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
 
    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    distribute_layers: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    n_batches: int | None = None 
    """Number of training batches. If None, it is automatically chosen based on the number of tokens."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1
    load_from_checkpoint: bool = False
    load_from_hub_path: str | None = None

    def __post_init__(self):
        assert not (self.layers and self.layer_stride != 1), "Cannot specify both `layers` and `layer_stride`."


@dataclass
class EvaluationTaskConfig(Serializable):
    """A task for evaluating the SAE."""
    task_fn: str
    dataset_path: str
    use_padding: bool
    chat_format: bool
    num_dataset_samples: int
    do_generations: bool
    max_seq_len: int

    def __post_init__(self):
        """Validate the task_fn after initialization."""
        valid_task_fns = ['evaluate_kl_div', 'evaluate_loss', 'generate_completions']
        if self.task_fn not in valid_task_fns:
            raise ValueError(f"task_fn must be one of {valid_task_fns}, but got {self.task_fn}")
    
    def to_path_string(self) -> str:
        """Convert the EvaluationTask to a string representation suitable for use as a path."""
        return (
            f"task_"
            f"{self.dataset_path.replace('/', '_')}_"
            f"pad_{self.use_padding}_"
            f"chat_{self.chat_format}_"
            f"samples_{self.num_dataset_samples}_"
            f"gen_{self.do_generations}_"
            f"seqlen_{self.max_seq_len}"
        ).replace('-', '_')

    def to_dict(self) -> Dict[str, Any]:
        """Convert the EvaluationTask to a dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "use_padding": self.use_padding,
            "chat_format": self.chat_format,
            "num_dataset_samples": self.num_dataset_samples,
            "do_generations": self.do_generations,
            "max_seq_len": self.max_seq_len
        }


@dataclass
class EvaluationTaskConfig(Serializable):
    """A task for evaluating the SAE."""
    task_fn: str
    dataset_path: str
    use_padding: bool
    chat_format: bool
    num_dataset_samples: int
    do_generations: bool
    max_seq_len: int

    def __post_init__(self):
        """Validate the task_fn after initialization."""
        valid_task_fns = ['evaluate_kl_div', 'evaluate_loss', 'generate_completions']
        if self.task_fn not in valid_task_fns:
            raise ValueError(f"task_fn must be one of {valid_task_fns}, but got {self.task_fn}")
    
    def to_path_string(self) -> str:
        """Convert the EvaluationTask to a string representation suitable for use as a path."""
        return (
            f"task_"
            f"{self.dataset_path.replace('/', '_')}_"
            f"pad_{self.use_padding}_"
            f"chat_{self.chat_format}_"
            f"samples_{self.num_dataset_samples}_"
            f"gen_{self.do_generations}_"
            f"seqlen_{self.max_seq_len}"
        ).replace('-', '_')

    def to_dict(self) -> Dict[str, Any]:
        """Convert the EvaluationTask to a dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "use_padding": self.use_padding,
            "chat_format": self.chat_format,
            "num_dataset_samples": self.num_dataset_samples,
            "do_generations": self.do_generations,
            "max_seq_len": self.max_seq_len
        }

