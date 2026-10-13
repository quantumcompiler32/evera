"""Regression checks using temporary datasets; no cameras or ML dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from vision_dashboard import server


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in {
            'ROOT': self.root, 'CAPTURES': self.root / 'captures',
            'LABELS': self.root / 'labels', 'OUTPUTS': self.root / 'outputs',
            'ROLES_FILE': self.root / 'roles.json',
        }.items():
            patcher = patch.object(server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        base = self.root / 'datasets/yolo-seg/workshop-provisional-clean'
        base.mkdir(parents=True)
        for split in ('train', 'val'):
            (base / f'{split}.txt').write_text(f'/legacy/{split}.jpg\n')

    def frame(self, session, split, reviewed=True):
        filename = '20261011T123456123456Z.jpg'
        folder = server.CAPTURES / session
        folder.mkdir(parents=True)
        (folder / filename).write_bytes(b'fixture copied without image decoding')
        server.write_json(server.LABELS / session / f'{Path(filename).stem}.json', {
            'reviewed': reviewed,
            'polygons': [{'class': 'multimeter', 'points': [[0, 0], [1, 0], [1, 1]]}],
        })
        roles = server.read_json(server.ROLES_FILE, {})
        roles[session] = split
        server.write_json(server.ROLES_FILE, roles)

    def test_evaluation_only_new_object_does_not_enable_training(self):
        self.frame('meter-test', 'test')
        with self.assertRaisesRegex(ValueError, 'training frames'):
            server.build_dataset('run-test')

    def test_training_and_test_sessions_remain_separate(self):
        self.frame('meter-train', 'train')
        self.frame('meter-test', 'test')
        yaml, counts = server.build_dataset('run-valid')
        train = (yaml.parent / 'train.txt').read_text()
        test = (yaml.parent / 'test.txt').read_text()
        self.assertIn('meter-train', train)
        self.assertNotIn('meter-test', train)
        self.assertIn('meter-test', test)
        self.assertNotIn('meter-train', test)
        self.assertEqual(counts['multimeter'], 2)

    def test_unreviewed_manual_labels_are_excluded(self):
        self.frame('meter-train', 'train', reviewed=False)
        with self.assertRaisesRegex(ValueError, 'training frames'):
            server.build_dataset('run-unreviewed')

    def test_capture_success_does_not_claim_trained_model(self):
        log = self.root / 'capture.log'
        log.write_text('Saved frames\n')
        with patch.object(server, 'job', {'process': Mock(poll=Mock(return_value=0)),
                                         'kind': 'capture', 'log': str(log), 'output': ''}):
            self.assertIn('Capture complete', server.job_state()['summary'])
            self.assertNotIn('model ready', server.job_state()['summary'])


if __name__ == '__main__':
    unittest.main()
