import unittest,importlib.util
spec=importlib.util.spec_from_file_location('controller','reports/overnight_controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class T(unittest.TestCase):
 def state(self):return dict(total=0,cuda_retries={},nan_batches=[],memory_fraction=.4)
 def test_same_batch(self):
  s=self.state();self.assertTrue(m.decide(s,'cuda_memory',57)[0]);self.assertFalse(m.decide(s,'cuda_memory',57)[0])
 def test_total(self):
  s=self.state();s['total']=5;self.assertFalse(m.decide(s,'cuda_memory',57)[0])
 def test_nan_window(self):
  s=self.state();s['nan_batches']=[50,52];self.assertFalse(m.decide(s,'nonfinite',57)[0])
 def test_cuda_limit(self):
  s=self.state();s['cuda_retries']['cuda_memory']=2;self.assertFalse(m.decide(s,'cuda_memory',57)[0])
 def test_oom(self):
  s=self.state();self.assertTrue(m.decide(s,'oom',57)[0]);s['memory_fraction']=.35;self.assertFalse(m.decide(s,'oom',58)[0])
 def test_unknown(self):self.assertFalse(m.decide(self.state(),'unknown',57)[0])
 def test_classify(self):
  for text,kind in [('CUDA out of memory','oom'),('nonfinite gradient','nonfinite'),('CUDA error: an illegal memory access','cuda_memory'),('Below 12GiB free','disk')]:self.assertEqual(m.classify(text),kind)
if __name__=='__main__':unittest.main()
