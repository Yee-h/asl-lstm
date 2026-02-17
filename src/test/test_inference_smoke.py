import os
import sys
import unittest
from collections import deque

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.data_process.preprocess_wlasl import PreprocessHelper
from src.model.model_lstm import get_model
from src.model.realtime_inference import prepare_sequence


class TestInferenceSmoke(unittest.TestCase):
    def test_offline_frame_sequence_inference(self):
        frame_buffer = deque(maxlen=cfg.SEQUENCE.max_frames)
        for frame_idx in range(8):
            frame = np.zeros((2, cfg.SEQUENCE.num_landmarks), dtype=np.float32)
            frame[:, :10] = 0.01 * (frame_idx + 1)
            frame_buffer.append(frame)

        helper = PreprocessHelper(max_frames=cfg.SEQUENCE.max_frames)
        inputs, lengths = prepare_sequence(frame_buffer, helper, stats=None)

        model = get_model(use_attention=cfg.MODEL.use_attention)
        model.eval()

        with torch.no_grad():
            logits = model(inputs, lengths)

        self.assertEqual(tuple(inputs.shape), (1, cfg.SEQUENCE.max_frames, cfg.SEQUENCE.input_size))
        self.assertEqual(tuple(lengths.shape), (1,))
        self.assertEqual(tuple(logits.shape), (1, cfg.SEQUENCE.num_classes))


if __name__ == "__main__":
    unittest.main()
