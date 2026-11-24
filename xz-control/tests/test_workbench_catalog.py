import csv
import json
import tempfile
import unittest
from pathlib import Path
from xz_workbench import recording


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'data'; self.prefs=Path(self.tmp.name)/'prefs.json'
    def catalog(self):
        from xz_workbench.catalog import Catalog
        return Catalog(self.root,self.prefs)
    def test_catalog_api_exists(self):
        self.assertTrue(hasattr(recording,'CSV_HEADER'))
        from xz_workbench import catalog
        self.assertTrue(callable(catalog.Catalog))
    def test_sequences_reuse_free_names_and_per_label(self):
        c=self.catalog(); a=c.reserve('棉布'); self.assertEqual(a['run_name'],'棉布_001')
        a['path'].rmdir()
        self.assertEqual(self.catalog().reserve('棉布')['run_name'],'棉布_001')
        self.assertEqual(c.reserve('丝绸')['run_name'],'丝绸_001')
        self.assertEqual(self.catalog().last_label(),'丝绸')
    def test_existing_name_skipped_and_named_recorder_keeps_header(self):
        c=self.catalog(); self.root.mkdir(); (self.root/'A_001').mkdir()
        allocation=c.reserve('A'); r=recording.Recorder(self.root,(1,2,3),{},allocation=allocation)
        r.append(recording.Frame(1,1,0,r.start_ns,2,4,6),'manual_idle',{})
        path=r.close(); meta=json.loads((path/'metadata.json').read_text())
        self.assertEqual(path.name,'A_002'); self.assertEqual(meta['label'],'A')
        with (path/meta['files']['sensor']).open(encoding='utf-8-sig') as f:
            rows=list(csv.reader(f))
        self.assertEqual(rows[0],recording.CSV_HEADER); self.assertEqual(len(rows[1]),23)
    def legacy(self,outcome='completed'):
        path=self.root/'20261112_185152'; path.mkdir(parents=True)
        (path/'sensor.csv').write_bytes(b'original-csv\r\n')
        (path/'metadata.json').write_text(json.dumps({'outcome':outcome,'files':{'sensor':'sensor.csv'},'sample_count':5}))
        return path
    def test_legacy_rename_preserves_csv_and_updates_last_label(self):
        path=self.legacy(); target=self.catalog().rename(path,'亚麻')
        self.assertEqual(target.name,'亚麻_001'); self.assertFalse(path.exists())
        self.assertEqual((target/'亚麻_001.csv').read_bytes(),b'original-csv\r\n')
        meta=json.loads((target/'metadata.json').read_text()); self.assertEqual(meta['files']['sensor'],'亚麻_001.csv')
        self.assertEqual(self.catalog().last_label(),'亚麻')
        self.assertEqual(self.catalog().records()[0]['label'],'亚麻')
    def test_busy_or_interrupted_record_cannot_be_renamed(self):
        path=self.legacy()
        with self.assertRaises(ValueError): self.catalog().rename(path,'B',blocked=[path])
        meta=json.loads((path/'metadata.json').read_text()); meta['outcome']='interrupted'
        (path/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError): self.catalog().rename(path,'B')
        self.assertTrue((path/'sensor.csv').exists())
    def test_metadata_write_failure_rolls_back_names_and_preserves_data(self):
        from unittest.mock import patch
        from xz_workbench.catalog import atomic_json
        path=self.legacy(); failed=False
        def fail_metadata_once(target,value):
            nonlocal failed
            if Path(target).name=='metadata.json' and not failed:
                failed=True; raise OSError('disk full')
            return atomic_json(target,value)
        with patch('xz_workbench.catalog.atomic_json',side_effect=fail_metadata_once):
            with self.assertRaises(OSError): self.catalog().rename(path,'B')
        self.assertTrue(path.exists()); self.assertEqual((path/'sensor.csv').read_bytes(),b'original-csv\r\n')
        self.assertEqual(json.loads((path/'metadata.json').read_text())['files']['sensor'],'sensor.csv')
        self.assertEqual(self.catalog().next_name('B'),'B_001')

    def test_label_cannot_escape_directory(self):
        for label in ('','../data','a/b','a\\b','..','a\nB'):
            with self.subTest(label=label),self.assertRaises(ValueError): self.catalog().reserve(label)


if __name__=='__main__': unittest.main()

class RecycleTests(unittest.TestCase):
    setUp=CatalogTests.setUp
    catalog=CatalogTests.catalog
    legacy=CatalogTests.legacy
    def test_deleted_gap_is_reused_and_restore_conflict_preserves_csv(self):
        c=self.catalog(); path=c.rename(self.legacy(),'棉布')
        second=c.reserve('棉布'); second['path'].rmdir()
        recycled=c.trash([path])
        self.assertFalse(path.exists()); self.assertEqual(c.next_name('棉布'),'棉布_001')
        reused=c.reserve('棉布'); self.assertEqual(reused['run_name'],'棉布_001')
        restored=c.restore(recycled[0])
        self.assertEqual(restored.name,'棉布_002')
        self.assertEqual((restored/'棉布_002.csv').read_bytes(),b'original-csv\r\n')
        self.assertEqual(c.trash_records(),[])

    def test_multiselect_preflight_blocks_active_without_partial_delete(self):
        c=self.catalog(); a=c.rename(self.legacy(),'A'); b=c.rename(self.legacy(),'B')
        with self.assertRaises(ValueError): c.trash([a,b],blocked=[b])
        self.assertTrue(a.exists()); self.assertTrue(b.exists())

    def test_paths_and_links_cannot_delete_outside_root(self):
        c=self.catalog(); path=self.legacy(); outside=Path(self.tmp.name)/'outside'; outside.mkdir()
        link=self.root/'linked'; link.symlink_to(outside,target_is_directory=True)
        for candidate in [outside,link,self.root]:
            with self.subTest(candidate=candidate),self.assertRaises(ValueError): c.trash([candidate])
        (path/'sensor.csv').unlink(); (path/'sensor.csv').symlink_to(outside/'data.csv')
        with self.assertRaises(ValueError): c.trash([path])
        self.assertTrue(outside.exists())

    def test_rename_releases_old_name(self):
        c=self.catalog(); path=c.rename(self.legacy(),'A'); c.rename(path,'B')
        self.assertEqual(c.next_name('A'),'A_001')

class AtomicReceiptTests(unittest.TestCase):
    setUp=CatalogTests.setUp
    catalog=CatalogTests.catalog
    legacy=CatalogTests.legacy
    def test_receipt_temp_symlink_never_changes_external_file(self):
        path=self.legacy(); victim=Path(self.tmp.name)/'outside.txt'; victim.write_text('untouched')
        (path/'.trash-info.json.tmp').symlink_to(victim)
        self.catalog().trash([path])
        self.assertEqual(victim.read_text(),'untouched')
