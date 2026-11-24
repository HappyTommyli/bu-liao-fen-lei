import json
import tempfile
import unittest
from pathlib import Path

class PreferenceTests(unittest.TestCase):
    def test_session_save_merges_existing_label_and_invalid_json_is_reported(self):
        from xz_workbench.preferences import update_settings,load_settings
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'settings.json'
            update_settings(path,{'last_label':'棉布'})
            update_settings(path,{'session':{'distance':2.5}})
            self.assertEqual(load_settings(path),{'last_label':'棉布','session':{'distance':2.5}})
            path.write_text('{bad')
            with self.assertRaises(ValueError): load_settings(path)
