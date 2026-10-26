"""Prevent accidental re-execution of scientific stages during remote continuation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('resume', Path(__file__).with_name('continue_four.py'))
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class StageEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = resume.ROOT
        resume.ROOT = Path(self.temp.name)
        self.arm = resume.ROOT / 'loop-carry-adam'
        self.arm.mkdir()

    def tearDown(self):
        resume.ROOT = self.old
        self.temp.cleanup()

    def put(self, path, data):
        target = self.arm / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))

    def test_unstarted(self):
        self.assertFalse(resume.existing('loop-carry-adam', 'science'))

    def test_complete_is_reused(self):
        self.put('launch-science.json', {'source_sha256': resume.MANIFEST})
        self.put('exit-science.json', {'ExitCode': 0})
        self.put('output/science/result.json', {'passed': True})
        self.assertTrue(resume.existing('loop-carry-adam', 'science'))

    def test_partial_launch_cannot_repeat(self):
        self.put('launch-science.json', {'source_sha256': resume.MANIFEST})
        with self.assertRaises(AssertionError):
            resume.existing('loop-carry-adam', 'science')

    def test_failed_science_cannot_repeat(self):
        self.put('output/science/failure.json', {'error': 'interrupted'})
        with self.assertRaises(RuntimeError):
            resume.existing('loop-carry-adam', 'science')

    def test_nonzero_exit_cannot_reuse(self):
        self.put('launch-science.json', {'source_sha256': resume.MANIFEST})
        self.put('exit-science.json', {'ExitCode': 1})
        self.put('output/science/result.json', {'passed': True})
        with self.assertRaises(AssertionError):
            resume.existing('loop-carry-adam', 'science')


if __name__ == '__main__':
    unittest.main()
