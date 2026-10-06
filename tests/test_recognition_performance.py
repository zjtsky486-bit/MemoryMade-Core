import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from memorymade import ai_stack
from memorymade.recognition_cache import RecognitionCache


class RecognitionPerformanceTests(unittest.TestCase):
    def test_copied_upload_reuses_full_result_without_job_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            first=Path(folder)/'a.jpg'; second=Path(folder)/'b.jpg'
            first.write_bytes(b'same photo'); second.write_bytes(first.read_bytes())
            report={'ok':True,'errors':[],'description':{'visible_parts':['wheel']}}
            with patch.object(ai_stack,'ANALYSIS_CACHE',RecognitionCache()), \
                 patch.object(ai_stack,'stack_status',return_value={'qwen':True,'vision_core':'test'}), \
                 patch.object(ai_stack,'_analysis_signature',return_value='v1'), \
                 patch.object(ai_stack,'run_worker',return_value=report) as worker:
                result=ai_stack.analyze_object(first)
                result['description']['visible_parts'].append('job-only')
                result['segmentation']={'foreground':'old job'}
                copied=ai_stack.analyze_object(second)
                self.assertEqual(worker.call_count,1)
                self.assertEqual(copied,report)

    def test_changed_pixels_target_and_model_signature_invalidate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.jpg';path.write_bytes(b'photo one')
            with patch.object(ai_stack,'ANALYSIS_CACHE',RecognitionCache()), \
                 patch.object(ai_stack,'stack_status',return_value={'qwen':True,'vision_core':'test'}), \
                 patch.object(ai_stack,'_analysis_signature',return_value='v1') as signature, \
                 patch.object(ai_stack,'run_worker',return_value={'ok':True}) as worker:
                ai_stack.analyze_object(path)
                path.write_bytes(b'photo two')
                ai_stack.analyze_object(path)
                ai_stack.analyze_object(path,'look at carving')
                signature.return_value='v2'
                ai_stack.analyze_object(path,'look at carving')
                self.assertEqual(worker.call_count,4)

    def test_concurrent_duplicate_upload_runs_one_inference(self):
        cache=RecognitionCache();entered=threading.Event();release=threading.Event()
        calls=[]
        def compute():
            calls.append(1);entered.set()
            if not release.wait(5):raise TimeoutError()
            return {'ok':True,'description':{'parts':[]}}
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first=pool.submit(cache.get,'image',compute)
            self.assertTrue(entered.wait(5))
            second=pool.submit(cache.get,'image',compute)
            release.set()
            a=first.result(5);b=second.result(5)
        a['description']['parts'].append('mutated')
        self.assertEqual(b['description']['parts'],[])
        self.assertEqual(len(calls),1)

    def test_partial_failures_are_retried_without_dropping_evidence(self):
        for report in ({'ok':False}, {'ok':True,'errors':['detector failed']},
                       {'ok':True,'description':{'fallback_reason':'temporary'}},
                       {'ok':True,'description':{'detail_review':{'status':'incomplete'}}}):
            with self.subTest(report=report):
                cache=RecognitionCache()
                calls=[]
                def compute():calls.append(1);return report
                self.assertEqual(cache.get('a',compute),report)
                cache.get('a',compute)
                self.assertEqual(len(calls),2)

    def test_exception_does_not_poison_future_requests(self):
        cache=RecognitionCache()
        with self.assertRaises(RuntimeError):
            cache.get('a',lambda:(_ for _ in ()).throw(RuntimeError('transient')))
        self.assertEqual(cache.get('a',lambda:{'ok':True}),{'ok':True})

    def test_cache_is_bounded(self):
        cache=RecognitionCache(2);calls=[]
        def compute():calls.append(1);return {'ok':True}
        for key in ('a','b','c','a'):cache.get(key,compute)
        self.assertEqual(len(calls),4)

    def test_disk_guard_applies_even_to_cached_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.jpg';path.write_bytes(b'photo')
            with patch.object(ai_stack,'ANALYSIS_CACHE',RecognitionCache()), \
                 patch.object(ai_stack,'stack_status',return_value={'qwen':True,'vision_core':'test'}), \
                 patch.object(ai_stack,'_analysis_signature',return_value='v1'), \
                 patch.object(ai_stack,'run_worker',return_value={'ok':True}) as worker, \
                 patch.object(ai_stack,'ensure_disk_space') as guard:
                self.assertTrue(ai_stack.analyze_object(path)['ok'])
                guard.side_effect=RuntimeError('disk reserve reached')
                self.assertFalse(ai_stack.analyze_object(path)['ok'])
                self.assertEqual(worker.call_count,1)

    def test_photo_changed_during_inference_is_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.jpg';path.write_bytes(b'photo')
            def changing_worker(*args,**kwargs):
                path.write_bytes(b'different photo');return {'ok':True}
            with patch.object(ai_stack,'ANALYSIS_CACHE',RecognitionCache()), \
                 patch.object(ai_stack,'stack_status',return_value={'qwen':True,'vision_core':'test'}), \
                 patch.object(ai_stack,'_analysis_signature',return_value='v1'), \
                 patch.object(ai_stack,'run_worker',side_effect=changing_worker) as worker:
                self.assertFalse(ai_stack.analyze_object(path)['ok'])
                worker.side_effect=None;worker.return_value={'ok':True}
                self.assertTrue(ai_stack.analyze_object(path)['ok'])
                self.assertEqual(worker.call_count,2)


if __name__=='__main__':unittest.main()
