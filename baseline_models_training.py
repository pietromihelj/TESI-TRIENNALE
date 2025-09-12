from Models_tr_aux_functions import train_fast_ICA, train_KPCA, train_PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, required=True, help='in_dir')
parser.add_argument('--out_dir', type=str, required=True, help='out_dir')
parser.add_argument('--clip_len', type=int, required=True, help='clip_len')

print('DEBUG: argomenti passati correttamente')

otps = parser.parse_args()

in_dir = otps.in_dir
out_dir = otps.out_dir
clip_len = otps.clip_len

start_time = 0
end_time = 250

"""
print('DEBUG: inizio training fast-ica')
alg = ['parallel', 'deflation']
fun = ['logcosh', 'exp', 'cube']
for a in alg:
    for f in fun:
        print('DEBUG: training fastica con: ', a, ' ', f)
        train_fast_ICA(in_dir=in_dir, out_dir=out_dir, start_time=start_time, end_time=end_time, alghoritm=a, fun=f)

print('DEBUG: inizio train pca')
train_PCA(in_dir=in_dir, out_dir=out_dir, start_time=start_time, end_time=end_time)
"""

print('DEBUG: inizo train k-pca')
kernel = [ 'rbf', 'poly', 'sigmoid']
args = [[0.01,0.1,1,5,10],[10,15,20,50,100],[0.01,0.1,1,5,10]]
for k,a in zip(kernel, args):
    for arg in a:
        for i in range(13000, 13500, 100):
        
            print('DEBUG: training kpca con: ', k,' ', arg,' ',i)
            train_KPCA(in_dir=in_dir, out_dir=out_dir, start_time=start_time, end_time=end_time, kernel=k, alpha=arg, n_comp=i)
        
        
print('DONE!')