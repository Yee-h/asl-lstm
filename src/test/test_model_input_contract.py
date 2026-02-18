import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.model.model_lstm import BiLSTMAttention, build_dummy_batch_for_smoke


class TestModelInputContract(unittest.TestCase):
    def test_build_dummy_batch_uses_config_input_size(self):
        inputs, lengths = build_dummy_batch_for_smoke(batch_size=4, seq_len=17)

        self.assertEqual(tuple(inputs.shape), (4, 17, cfg.SEQUENCE.input_size))
        self.assertEqual(tuple(lengths.shape), (4,))

    def test_dummy_batch_can_forward_without_shape_error(self):
        model = BiLSTMAttention()
        inputs, lengths = build_dummy_batch_for_smoke(batch_size=2, seq_len=11)

        logits = model(inputs, lengths)
        self.assertEqual(tuple(logits.shape), (2, cfg.SEQUENCE.num_classes))

    def test_attention_model_parameter_budget_is_controlled(self):
        model = BiLSTMAttention()
        total_params = sum(p.numel() for p in model.parameters())

        # WLASL100 小样本场景下，参数过大容易出现训练/验证分化。
        self.assertLessEqual(total_params, 1_300_000)


if __name__ == "__main__":
    unittest.main()
