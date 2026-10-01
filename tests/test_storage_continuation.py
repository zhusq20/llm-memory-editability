"""Continuation-only exposures include the end of one shuffled epoch."""

import numpy as np

from llm_memory_editability.grok_depth import EpochStream


def test_partial_epoch_continuation_restores_sequence_and_requires_range_two():
    stream = EpochStream(768, 41)
    stream.take(16000 * 64)
    restored = EpochStream(768, 0)
    restored.load_state_dict(stream.state_dict())
    drawn = stream.take(4096 * 64)
    np.testing.assert_array_equal(drawn, restored.take(4096 * 64))
    counts = np.bincount(drawn, minlength=768)
    assert counts.sum() == 4096 * 64
    assert np.ptp(counts) == 2
