"""Actual-evaluator regression/integration tests; only temporary synthetic Dev ZIPs."""
from __future__ import annotations
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

import run_official_dev as current
from recovery_runtime import canonical_bytes, digest, validate_thresholds, validate_manifest_digest
from recheck_saved_results import recheck_saved_results

EVALUATOR = current
V1_SHA = "6deb2ace9f433173b483338172db07318a17a735"


class RecordingVectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.inputs = []
    def transform(self, texts):
        self.inputs.append(list(texts))
        return self.vectorizer.transform(texts)
    def fit(self, *args, **kwargs):
        raise AssertionError("evaluation must never fit")
    fit_transform = fit


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.v1 = current.read_json(current.FREEZE_PATH)
        self.items = [
            {"sentence1": "  Cafe\u0301  不退款 12! A  ", "sentence2": "Café 不退款 12! A", "label": "1"},
            {"sentence1": "订单  12 元", "sentence2": "订单 13 元", "label": "0"},
            {"sentence1": "ABC 不付款!", "sentence2": "abc 付款?", "label": "0"},
            {"sentence1": " 可以退款 ", "sentence2": "可以退款", "label": "1"},
        ]
        self.raw = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in self.items)
        self.archive = self.base / "synthetic.zip"
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr("dev.json", self.raw)
        self.records = current.parse_jsonl_records(self.raw, "dev")
        self.manifest = self.base / "manifest.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["source_file", "source_row_id", "official_split", "label"])
            w.writeheader()
            for row in self.records:
                w.writerow({"source_file": "dev.json", "source_row_id": row["source_row_id"], "official_split": "dev", "label": row["label"]})
        self.freeze = copy.deepcopy(self.v1)
        self.freeze["inputs"].update(archive_sha256=digest(self.archive.read_bytes()), dev_member_sha256=digest(self.raw),
                                    manifest_sha256=digest(self.manifest.read_bytes()), dev_rows=4)
        self.rehash()
        _, self.left, self.right = current.normalized_pair_texts(self.records)
        a, b, _, _ = current.fit_vectorizers(sorted(set(self.left + self.right)))
        self.vectorizers = {"count_char2": RecordingVectorizer(a), "tfidf_char2": RecordingVectorizer(b)}
        original_open = zipfile.ZipFile.open
        real_archive = (current.ROOT / "stage01_data_audit/raw/afqmc_public.zip").resolve()
        def guarded_open(archive, name, *args, **kwargs):
            if name == "dev.json" and Path(archive.filename).resolve() == real_archive:
                raise AssertionError("real AFQMC Dev is forbidden during tests")
            return original_open(archive, name, *args, **kwargs)
        self.guard = patch.object(zipfile.ZipFile, "open", guarded_open)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def rehash(self):
        self.freeze.pop("freeze_manifest_sha256", None)
        self.freeze["freeze_manifest_sha256"] = digest(canonical_bytes(self.freeze))

    def compute(self):
        return EVALUATOR.compute_run(self.freeze, self.vectorizers, self.archive, self.manifest)

    def test_actual_compute_accepts_nested_thresholds(self):
        summary, rows = self.compute()
        self.assertEqual(len(rows), 8)
        for method in current.METHODS:
            self.assertEqual(summary["metrics"][method]["threshold"].hex(), float(self.v1["thresholds"][method]["value"]).hex())

    def test_actual_compute_normalizes_like_stage3a(self):
        error = None
        try:
            summary, rows = self.compute()
        except TypeError as caught:
            error = caught
        for method, vectorizer in self.vectorizers.items():
            if vectorizer.inputs:
                self.assertEqual(vectorizer.inputs[0], self.left)
                self.assertEqual(vectorizer.inputs[1], self.right)
        if error:
            raise error
        for method, vectorizer in self.vectorizers.items():
            expected, _, _ = current.paired_sparse_cosine(vectorizer.vectorizer.transform(self.left), vectorizer.vectorizer.transform(self.right))
            actual = np.asarray([r["score"] for r in rows if r["method"] == method])
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.left[0], "Café 不退款 12! A")
        self.assertEqual(self.left[2], "ABC 不付款!")
        self.assertEqual(self.right[2], "abc 付款?")

    def test_threshold_full_precision(self):
        values = validate_thresholds(self.v1)
        self.assertEqual(values["count_char2"].hex(), (0.10660035817780521).hex())
        self.assertEqual(values["tfidf_char2"].hex(), (0.05574242073415668).hex())

    def test_invalid_thresholds_rejected_before_reader(self):
        entries = [None, True, False, "0.1", [], {}, float("nan"), float("inf"), -float("inf")]
        for method in current.METHODS:
            for value in entries:
                with self.subTest(method=method, value=type(value).__name__):
                    f = copy.deepcopy(self.freeze)
                    f["thresholds"][method]["value"] = value
                    with patch.object(current, "read_dev_records") as reader:
                        with self.assertRaises((ValueError, TypeError)):
                            current.compute_run(f, self.vectorizers, self.archive, self.manifest)
                        reader.assert_not_called()
            for entry in (None, 0.1, {"source": "missing value"}):
                f = copy.deepcopy(self.freeze); f["thresholds"][method] = entry
                with patch.object(current, "read_dev_records") as reader:
                    with self.assertRaises(ValueError):
                        current.compute_run(f, self.vectorizers, self.archive, self.manifest)
                    reader.assert_not_called()
        for thresholds in ({}, None, {"count_char2": self.freeze["thresholds"]["count_char2"]}):
            f = copy.deepcopy(self.freeze); f["thresholds"] = thresholds
            with patch.object(current, "read_dev_records") as reader:
                with self.assertRaises(ValueError): current.compute_run(f, self.vectorizers, self.archive, self.manifest)
                reader.assert_not_called()

    def test_fingerprint_format_and_content_block_reader(self):
        for bad in ("a" * 62, "g" * 64, "0" * 64):
            f = copy.deepcopy(self.freeze); f["freeze_manifest_sha256"] = bad
            with patch.object(current, "read_dev_records") as reader:
                with self.assertRaises((ValueError, AssertionError)):
                    current.compute_run(f, self.vectorizers, self.archive, self.manifest)
                reader.assert_not_called()
        f=copy.deepcopy(self.freeze); f["inputs"]["archive_sha256"]="0"*64
        f.pop("freeze_manifest_sha256"); f["freeze_manifest_sha256"]=digest(canonical_bytes(f))
        with patch.object(zipfile.ZipFile,"open") as opened:
            with self.assertRaises(AssertionError): current.compute_run(f,self.vectorizers,self.archive,self.manifest)
            opened.assert_not_called()

    def test_threshold_equality_is_positive_in_formal_function(self):
        equal_scores = [(np.full(4, self.v1["thresholds"][m]["value"]), np.zeros(4,dtype=bool),np.zeros(4,dtype=bool)) for m in current.METHODS]
        with patch.object(current, "paired_sparse_cosine", side_effect=equal_scores):
            _, rows = self.compute()
        self.assertTrue(all(r["prediction"] == 1 for r in rows))

    def test_all_methods_and_independent_confusions(self):
        summary, rows = self.compute()
        self.assertEqual(set(summary["metrics"]), set(current.METHODS))
        self.assertEqual(set(summary["constant_baselines"]), {"always_predict_0","always_predict_1"})
        y=np.asarray([r["label"] for r in self.records])
        for method, result in {**summary["metrics"], **summary["constant_baselines"]}.items():
            selected=[r for r in rows if r["method"]==method]
            p=np.asarray([r["prediction"] for r in selected]) if selected else np.full(4,int(method[-1]))
            tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
            self.assertEqual((tp,fp,fn,tn),(result["tp"],result["fp"],result["fn"],result["tn"]))
            self.assertEqual(float(f1_score(y,p,zero_division=0)),result["f1"])

    def test_json_csv_real_write_reread_and_recompute(self):
        output=self.base/'outputs'
        summary=current.run_saved_evaluation(self.freeze,self.vectorizers,self.archive,self.manifest,output,self.base/'attempts','attempt_002',{})
        loaded=json.loads((output/'dev_metrics.json').read_text())
        self.assertEqual(summary,loaded)
        rows=list(csv.DictReader(io.StringIO((output/'dev_predictions_local_only.csv').read_text())))
        metrics_csv=list(csv.DictReader(io.StringIO((output/'dev_metrics.csv').read_text())))
        self.assertEqual(len(rows),8); self.assertEqual(len(metrics_csv),4)
        for method in current.METHODS:
            selected=[r for r in rows if r['method']==method]
            y=np.asarray([int(r['label']) for r in selected]); p=np.asarray([int(r['prediction']) for r in selected]); s=np.asarray([float(r['score']) for r in selected])
            self.assertEqual(float(f1_score(y,p)),loaded['metrics'][method]['f1'])
            self.assertEqual(float(roc_auc_score(y,s)),loaded['metrics'][method]['roc_auc'])
            self.assertEqual(float(average_precision_score(y,s)),loaded['metrics'][method]['average_precision'])
        state=json.loads((self.base/'attempts/attempt_002/result.json').read_text())
        self.assertEqual(state['status'],'completed_recovery_evaluation')
        events=[json.loads(p.read_text()) for p in sorted((self.base/'attempts/attempt_002').glob('0*.json'))]
        stages=[e['stage'] for e in events]
        self.assertLess(stages.index('started'),stages.index('dev_read_started'))
        self.assertLess(stages.index('dev_read_completed'),stages.index('all_metrics_completed'))
        self.assertTrue(recheck_saved_results(output,self.freeze)['passed'])

    def test_frozen_source_mismatch_rejected_before_dev(self):
        v1=copy.deepcopy(self.freeze)
        stage=self.base/'stage03b_official_validation'; stage.mkdir()
        (stage/'pre_dev_freeze_manifest.json').write_bytes(canonical_bytes(v1))
        v2=copy.deepcopy(v1); v2['freeze_version']=2
        v2['recovery']={'v1_manifest_raw_bytes_sha256':digest((stage/'pre_dev_freeze_manifest.json').read_bytes()),'v1_manifest_canonical_sha256':v1['freeze_manifest_sha256']}
        (self.base/'source.py').write_text('modified source')
        v2['frozen_source_files_sha256']={'source.py':digest(b'original source')}
        v2.pop('freeze_manifest_sha256'); v2['freeze_manifest_sha256']=digest(canonical_bytes(v2))
        with patch.object(current,'ROOT',self.base), patch.object(current,'read_dev_records') as reader:
            with self.assertRaisesRegex(AssertionError,'source fingerprint'): current.verify_frozen_code(v2)
            reader.assert_not_called()

    def test_failed_attempt_persists_without_outputs_and_cannot_be_reused(self):
        output=self.base/'outputs'
        with patch.object(current,'paired_sparse_cosine',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError): current.run_saved_evaluation(self.freeze,self.vectorizers,self.archive,self.manifest,output,self.base/'attempts','attempt_002',{})
        state=json.loads((self.base/'attempts/attempt_002/result.json').read_text())
        self.assertTrue(state['dev_read_started']); self.assertFalse(output.exists())
        self.assertEqual(state['status'],'failed_after_partial_dev_access')
        with self.assertRaises(FileExistsError): current.run_saved_evaluation(self.freeze,self.vectorizers,self.archive,self.manifest,output,self.base/'attempts','attempt_002',{})

    def test_manifest_and_normalized_id_alignment_fail_closed(self):
        with patch.object(current,'normalized_pair_texts',return_value=([r['source_row_id'] for r in reversed(self.records)],self.left,self.right)):
            with self.assertRaises(AssertionError): self.compute()
        duplicate=copy.deepcopy(self.records); duplicate[1]['source_row_id']=duplicate[0]['source_row_id']
        with self.assertRaises(ValueError): current.normalized_pair_texts(duplicate)
        with patch.object(current,'parse_manifest_split',return_value=[{'source_row_id':r['source_row_id'],'label':str(r['label'])} for r in reversed(self.records)]):
            with self.assertRaises(AssertionError): self.compute()


def prove_v1_regressions():
    global EVALUATOR
    source=subprocess.check_output(['git','show',V1_SHA+':stage03b_official_validation/run_official_dev.py'],cwd=current.ROOT).decode('utf-8')
    module=types.ModuleType('original_v1_evaluator'); module.__file__=current.__file__
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    EVALUATOR=module
    suite=unittest.TestSuite(RecoveryTests(name) for name in ('test_actual_compute_accepts_nested_thresholds','test_actual_compute_normalizes_like_stage3a'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    expected=len(result.errors)==1 and len(result.failures)==1 and result.testsRun==2
    print(json.dumps({'v1_threshold_error_reproduced':len(result.errors)==1,'v1_normalization_failure_reproduced':len(result.failures)==1,'expected_regressions_proven':expected}))
    return 0 if expected else 1


if __name__=='__main__':
    if '--prove-v1' in sys.argv: raise SystemExit(prove_v1_regressions())
    unittest.main(verbosity=2)
