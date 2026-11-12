"""Lossless raw-result storage must retain committed bytes and original logical hashes."""
import gzip,hashlib,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import render_pilot_report as report
class CompressedResults(unittest.TestCase):
    def test_compressed_only_checkout_verifies_original_bytes_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);subprocess.run(['git','init','-q',directory],check=True)
            raw=root/'paper/results/run/predictions.jsonl';raw.parent.mkdir(parents=True)
            packed=Path(str(raw)+'.gz');payload=b'{"value": 7}\n';packed.write_bytes(gzip.compress(payload,mtime=0))
            subprocess.run(['git','add','.'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=t@example.invalid','commit','-qm','fixture'],cwd=root,check=True)
            with patch.object(report,'__file__',str(root/'paper/scripts/render_pilot_report.py')):
                self.assertEqual(report.committed_json(raw,True),[{'value':7}])
                self.assertEqual(report.artifact_digest(raw),hashlib.sha256(payload).hexdigest())
                raw.write_bytes(b'{"value":8}\n')
                with self.assertRaises(ValueError):report.committed_json(raw,True)
                raw.unlink();packed.write_bytes(packed.read_bytes()+b'changed')
                with self.assertRaises(ValueError):report.committed_json(raw,True)
    def test_analysis_finishes_with_compressed_only_result_provenance(self):
        import json,io,tarfile
        import analyze_reachability as analysis
        from audit_competition_data import digest
        from localization_retrieval import delete_edges
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            (root/'tasks.jsonl').write_text(json.dumps(dict(instance_id='pkg_1',repo='o/pkg',base_commit='a'*40,problem_statement='fix'))+'\n')
            source=b'def f():\n    return 1\ndef g():\n    return 2\n'
            with tarfile.open(root/'snapshot.tgz','w:gz') as archive:
                entry=tarfile.TarInfo('x.py');entry.size=len(source);archive.addfile(entry,io.BytesIO(source))
            edges=[dict(source='x.f',target='x.g')]
            (root/'graph.json').write_text(json.dumps({'nodes':[{'id':'x.f'},{'id':'x.g'}],'edges':edges}))
            for name in ['audit.txt','pilot.txt','config.txt']:(root/name).write_text('{}')
            (run/'manifest.json').write_text('{}')
            payload=b'{}\n';(run/'per_task.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
            task=dict(instance_id='pkg_1',repo='o/pkg',graph_path='graph.json',snapshot_path='snapshot.tgz')
            audit={'assets':{name:{'sha256':digest(root/name)} for name in ['graph.json','snapshot.tgz']}}
            pilot={'tasks':[task],'tasks_sha256':digest(root/'tasks.jsonl')}
            config={'bm25':{'k1':1.2,'b':.75},'one_hop':{'seed_count':5}}
            clean=dict(instance_id='pkg_1',method='one_hop',severity=0,corruption='clean',seed=0,ranked_files=['x.py'],metrics={'recall@5':1})
            corrupt={**clean,'severity':.5,'corruption':'missing','seed':13,'intervention':delete_edges(edges,.5,13)[1]}
            args=['analyze_reachability.py','--data-root',str(root),'--run-dir',str(run),'--pilot',str(root/'pilot.txt'),'--audit',str(root/'audit.txt'),'--config',str(root/'config.txt'),'--mapping-dir',str(root/'mapping'),'--output',str(root/'output.txt')]
            with patch.object(sys,'argv',args),patch.object(analysis,'load_verified_run',return_value=(audit,pilot,{'configuration':config},[],[clean,corrupt],{})):
                analysis.main()
            output=json.loads((root/'output.txt').read_text())
            self.assertEqual(output['input_hashes'][str(run/'per_task.jsonl')],hashlib.sha256(payload).hexdigest())
