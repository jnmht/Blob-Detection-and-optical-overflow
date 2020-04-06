from utils import *
from functions_task1 import *

N_THRESHOLDS = 99


if __name__ == '__main__':
  start = time.time()
  imset = 'val'
  imlist = get_imlist(imset)
  output_dir = 'contour-output-task1/demo'; fn = compute_edges_dxdy_new;
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  print('Running detector:')
  detect_edges(imlist, fn, output_dir)
  
  _load_pred = lambda x: load_pred(output_dir, x)
  print('Evaluating:')
  sample_results, threshold_results, overall_result = \
    evaluate_boundaries.pr_evaluation(N_THRESHOLDS, imlist, load_gt_boundaries, 
                                      _load_pred, fast=True, progress=tqdm)
  fig = plt.figure(figsize=(6,6))
  ax = fig.gca()
  file_name = os.path.join(output_dir + '_out.txt')
  with open(file_name, 'wt') as f:
    display_results(ax, f, sample_results, threshold_results, overall_result)
  fig.savefig(os.path.join(output_dir + '_pr.pdf'), bbox_inches='tight')
  print(time.time()-start)
