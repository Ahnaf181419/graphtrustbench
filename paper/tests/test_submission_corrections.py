import json,re,sys,unittest,tempfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_reachability as a
import agent_readout as readout

class Corrections(unittest.TestCase):
    def test_distinct_outcomes(self):
        clean={'ranked_files':['g','a'], 'metrics':{'recall@5':1}}
        other={'ranked_files':['g','b'], 'metrics':{'recall@5':1}}
        self.assertEqual(a.outcome_changes(clean,other),dict(recall5_changed=False,top5_order_changed=True,top5_set_changed=True,full_ranking_changed=True))
        other['ranked_files']=['a','g']
        self.assertFalse(a.outcome_changes(clean,other)['top5_set_changed'])
    def test_correlation_imbalanced(self):
        self.assertAlmostEqual(a.point_biserial([1,0,0,0],[True,False,False,False]),1)
        self.assertAlmostEqual(a.point_biserial([0,1,1,1],[True,False,False,False]),-1)
        self.assertIsNone(a.point_biserial([1,2],[True,True]))
        with self.assertRaises(ValueError):a.point_biserial([1],[])
    def test_exact_selection_probability(self):
        self.assertAlmostEqual(a.zero_selection_probability(10,2,5),2/9)
        self.assertEqual(a.zero_selection_probability(10,0,5),1)
        self.assertEqual(a.zero_selection_probability(10,6,5),0)
    def test_source_budget(self):
        for n in [5,7]:
            files=[f'{i}.py' for i in range(n)]
            prompt=readout.build_prompt('issue',files,{f:'x'*3000 for f in files})
            empty=readout.PROMPT_TEMPLATE.format(issue='issue',files='')
            self.assertLessEqual(len(prompt)-len(empty),6000)
    def test_missing_source_fails(self):
        with self.assertRaises(ValueError):readout.build_prompt('issue',['missing.py'],{})
    def test_notebook_dry_run_never_imports_model_runtime(self):
        import json
        nb=json.loads((Path(__file__).resolve().parents[1]/'notebooks/agent_readout.ipynb').read_text())
        exec(''.join(nb['cells'][7]['source']),{'DRY_RUN':True})
    def test_prompt_all_candidates_remain_listed_with_tiny_budget(self):
        files=['a.py','b.py'];prompt=readout.build_prompt('issue',files,dict.fromkeys(files,'x'*500),total_chars=80)
        self.assertIn('a.py',prompt);self.assertIn('b.py',prompt)

    def test_prepare_contexts_uses_distinct_verified_snapshots_and_fails_closed(self):
        import json,io,tarfile,copy
        from audit_competition_data import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);items=[];tasks=[];assets={}
            for i in (1,2):
                snapshot=f'{i}.tgz';payload=f'SOURCE_FOR_TASK_{i}'.encode()
                with tarfile.open(root/snapshot,'w:gz') as archive:
                    entry=tarfile.TarInfo('same.py');entry.size=len(payload);archive.addfile(entry,io.BytesIO(payload))
                item=dict(instance_id=f'pkg_{i}',repo='o/pkg',base_commit=str(i)*40,snapshot_path=snapshot,graph_path='graph',embedding_path='embedding')
                items.append(item);tasks.append({**item,'problem_statement':'fix issue'})
                assets[snapshot]={'sha256':digest(root/snapshot)}
            assets.update(graph={},embedding={})
            (root/'tasks.jsonl').write_text('\n'.join(json.dumps(t) for t in tasks)+'\n')
            selection=dict(status='passed_schema_preflight',dataset_fingerprint='fixture',tasks_sha256=digest(root/'tasks.jsonl'),tasks=items)
            audit={**selection,'scan_snapshots':True,'tasks':[{**t,'schema_usable':True} for t in items],'assets':assets}
            pairs=[dict(instance_id=t['instance_id'],clean_files=['same.py'],corrupted_files=['same.py'],gold_files=['same.py']) for t in items]
            records=readout.prepare_contexts(root,selection,audit,pairs)
            self.assertIn('SOURCE_FOR_TASK_1',records[0]['prompt']);self.assertNotIn('SOURCE_FOR_TASK_2',records[0]['prompt'])
            self.assertIn('SOURCE_FOR_TASK_2',records[2]['prompt'])
            bad=copy.deepcopy(pairs);bad[0]['clean_files']=['absent.py']
            with self.assertRaisesRegex(ValueError,'Missing candidate'):readout.prepare_contexts(root,selection,audit,bad)
            bad=copy.deepcopy(pairs);bad[0]['instance_id']='unknown'
            with self.assertRaisesRegex(ValueError,'Unknown selected'):readout.prepare_contexts(root,selection,audit,bad)
            (root/'1.tgz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Snapshot fingerprint'):readout.prepare_contexts(root,selection,audit,pairs)
            (root/'1.tgz').unlink()
            with self.assertRaises(FileNotFoundError):readout.prepare_contexts(root,selection,audit,pairs)
