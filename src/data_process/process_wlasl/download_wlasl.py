from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CristianLazoQuispe/pose-action-recognition",
    repo_type="dataset",
    local_dir="pose_action_dataset",
    allow_patterns="*WLASL*",
)
