import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import model_bundle as bundle
from storage import require_output, work_dir


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.work=self.root/'medical-deid-rs';self.work.mkdir()
        self.environment=patch.dict(os.environ,{'DEID_STORAGE_ROOT':str(self.root),'DEID_WORK':str(self.work),'DEID_REQUIRE_EXTERNAL':'0'})
        self.environment.start()
        self.model=self.work/'base';self.model.mkdir()
        (self.model/'burn-config.json').write_text('{}')
        (self.model/'tokenizer.json').write_text('{}')
        self.checkpoint=self.work/'model.bpk';self.checkpoint.write_bytes(b'synthetic-checkpoint')
        self.card=self.work/'card.md';self.card.write_text('Synthetic fixture')

    def tearDown(self):
        self.environment.stop();self.temporary.cleanup()

    def pack(self):
        archive=self.work/'bundle.tar.gz'
        info=bundle.pack(self.checkpoint,self.model,self.card,archive,'test-model')
        return archive,info['sha256']

    def test_roundtrip_and_tampered_installed_file(self):
        archive,checksum=self.pack();destination=self.work/'installed'
        result=bundle.install(archive,destination,checksum)
        self.assertEqual(result['model_id'],'test-model')
        self.assertEqual(json.loads((destination/'labels.json').read_text()),bundle.LABELS)
        (destination/'best.bpk').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):bundle.verify(destination)

    def test_wrong_archive_checksum_does_not_create_destination(self):
        archive,_=self.pack();destination=self.work/'installed'
        with self.assertRaises(ValueError):bundle.install(archive,destination,'0'*64)
        self.assertFalse(destination.exists())

    def test_archive_rejects_traversal_links_and_duplicate_members(self):
        for index,kind in enumerate(['../escaped','symlink','duplicate']):
            archive=self.work/f'unsafe-{index}.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                item=tarfile.TarInfo('best.bpk' if kind != '../escaped' else kind)
                if kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname='../escaped'
                else:item.size=1
                tar.addfile(item,io.BytesIO(b'x') if item.isfile() else None)
                if kind=='duplicate':tar.addfile(item,io.BytesIO(b'x'))
            destination=self.work/f'installed-{index}'
            with self.assertRaises(ValueError):bundle.install(archive,destination,bundle.digest(archive))
            self.assertFalse(destination.exists())
        self.assertFalse((self.work/'escaped').exists())

    def test_output_outside_workspace_and_symlink_escape(self):
        with self.assertRaises(ValueError):require_output(self.root/'outside')
        (self.work/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):require_output(self.work/'link/file')

    def test_missing_external_volume_and_workspace_symlink(self):
        with patch.dict(os.environ,{'DEID_REQUIRE_EXTERNAL':'1'}):
            with self.assertRaises(ValueError):work_dir()
        alternate=self.root/'alternate';alternate.mkdir()
        outside=self.root/'outside';outside.mkdir()
        link=alternate/'medical-deid-rs';link.symlink_to(outside,target_is_directory=True)
        with patch.dict(os.environ,{'DEID_STORAGE_ROOT':str(alternate),'DEID_WORK':str(link)}):
            with self.assertRaises(ValueError):work_dir()


if __name__=='__main__':unittest.main()
