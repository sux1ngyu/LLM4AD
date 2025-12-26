from __future__ import annotations

from pathlib import PurePosixPath
from huggingface_hub import list_repo_files
from datasets import load_dataset


def select_indices_by_split(
    total: int,
    *,
    split: str,
    dev_indices: list[int] | None,
) -> list[int]:
    """
    Select instance indices according to split semantics (mirrors CO-Bench).

    - split="all": all indices [0..total-1]
    - split="dev": indices in dev_indices (if provided), else all
    - split="test": indices not in dev_indices (if provided), else all
    
    Note: If dev_indices is an empty list [], it is treated as [0] (CO-Bench semantics).
    """
    if total <= 0:
        return []

    split = (split or "all").lower()
    if split not in ("all", "dev", "test"):
        raise ValueError(f"Unknown split='{split}'. Expected one of: all/dev/test.")

    all_idx = list(range(total))
    if split == "all":
        return all_idx

    if dev_indices is None:
        # Same as CO-Bench: if no dev mapping exists, dev/test collapse to all.
        return all_idx

    # CO-Bench semantics: empty list [] is treated as [0]
    if len(dev_indices) == 0:
        dev_indices = [0]

    dev_set = {i for i in dev_indices if 0 <= i < total}
    if split == "dev":
        return sorted(dev_set)
    # split == "test"
    return [i for i in all_idx if i not in dev_set]


def load_subdir_as_text(repo_id: str, subdir: str, *, skip_ext: tuple[str, ...] = (".py",), streaming: bool = False):
    """
    Load files from a subdirectory in a Hugging Face dataset as text format.
    
    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        skip_ext: File extensions to skip (default: (".py",))
        streaming: Whether to use streaming mode
        
    Returns:
        A dict where keys are original filenames and values are loaded datasets
        
    Example:
        ds = load_subdir_as_text("CO-Bench/CO-Bench", "Aircraft landing")
        # Returns: {"airland1.txt": Dataset(...), "airland2.txt": Dataset(...), ...}
    """
    prefix = subdir.rstrip("/") + "/"
    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and not f.endswith(skip_ext)
    ]
    if not files:
        raise FileNotFoundError(f"No matching files inside '{subdir}' on {repo_id}")
    
    # Create a mapping from sanitized split names to original filenames
    def sanitize_split_name(filename):
        """Convert filename to valid split name (only alphanumeric, dots, underscores)"""
        import re
        # Replace hyphens and other special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._]', '_', filename)
        return sanitized
    
    # Build data_files dict with sanitized split names
    data_files = {}
    filename_mapping = {}  # Maps sanitized names back to original names
    
    for f in files:
        original_filename = PurePosixPath(f).name
        sanitized_name = sanitize_split_name(original_filename)
        data_files[sanitized_name] = f
        filename_mapping[sanitized_name] = original_filename
    
    # Load the dataset
    dataset = load_dataset(
        repo_id,
        data_files=data_files,
        streaming=streaming,
    )
    
    # Return a dict with original filenames as keys
    result = {}
    for sanitized_name, original_filename in filename_mapping.items():
        result[original_filename] = dataset[sanitized_name]
    
    return result


def load_subdir_as_pickle(repo_id: str, subdir: str, *, include_subdirs: tuple[str, ...] = (), streaming: bool = False):
    """
    Load pickle files from a subdirectory in a Hugging Face dataset.
    
    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        include_subdirs: Tuple of subdirectory names to include (if empty, includes all)
        streaming: Whether to use streaming mode
        
    Returns:
        A dict where keys are subdirectory names and values are dicts of 
        {filename: loaded_pickle_content}
        
    Example:
        result = load_subdir_as_pickle("CO-Bench/CO-Bench", "Maximal independent set", 
                                     include_subdirs=("er_test", "er_large_test"))
        # Returns: {"er_test": {"file1.gpickle": graph1, ...}, "er_large_test": {...}}
    """
    import pickle
    from huggingface_hub import hf_hub_download
    
    prefix = subdir.rstrip("/") + "/"
    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(('.pickle', '.gpickle', '.pkl'))
    ]
    
    if not files:
        raise FileNotFoundError(f"No pickle files found inside '{subdir}' on {repo_id}")
    
    # Organize files by subdirectory
    subdirs = {}
    for file_path in files:
        parts = file_path.split('/')
        if len(parts) >= 3:  # "subdir/subsubdir/filename"
            subsubdir = parts[1]  # The subdirectory under main subdir
            filename = parts[2]   # The actual filename
            
            # Filter by include_subdirs if specified
            if include_subdirs and subsubdir not in include_subdirs:
                continue
                
            if subsubdir not in subdirs:
                subdirs[subsubdir] = {}
            
            # Download and load the pickle file
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset"
                )
                
                with open(local_path, "rb") as f:
                    pickle_content = pickle.load(f)
                
                subdirs[subsubdir][filename] = pickle_content
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    return subdirs 