from dataset.dataset_ESC50 import InMemoryESC50
import config

testset = InMemoryESC50(subset="train", root=config.esc50_path, download=True)