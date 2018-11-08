import time
import random
import numpy as np
from algorithms.gru4rec import gru4rec2
from evaluation import evaluation as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac
from evaluation.metrics import coverage as cov
from evaluation.metrics import popularity as pop
import theano
import theano.misc.pkl_utils as pkl
import sys
import traceback
import gc


if __name__ == '__main__':
    
    '''
    Configuration
    '''
    data_path = 'data/rsc15/single/'
    file_prefix = 'rsc15-clicks'

    limit_train = None  # limit in number of rows or None
    limit_test = None  # limit in number of rows or None
    density_value = 1  # randomly filter out events (0.0-1.0, 1:keep all)
    
    export_csv_base = 'results/run-test-rsc15-opt'
    
    learn_rates = np.linspace(0.05, 0.8, num=16).astype( theano.config.floatX )
    drop_outs = np.linspace(0.0, 0.9, num=10).astype( theano.config.floatX )
    momentums = np.linspace(0.0, 0.9, num=10).astype( theano.config.floatX )
    losses = ['bpr-max-0.5', 'top1-max']
    
    best = 0.0
    best_key = ''
    
    for test_num in range(33):
        
        try: 
        
            print('process test ' + str(test_num))
            
            lr = random.choice(learn_rates)
            drop = random.choice(drop_outs)
            momentum = random.choice(momentums)
            loss = random.choice(losses)
            
            export_csv = export_csv_base + str(test_num) + '.csv'
        
            # create a list of metric classes to be evaluated
            metric = []
            metric.append(ac.HitRate(20))
            metric.append(ac.HitRate(10))
            metric.append(ac.HitRate(3))
            metric.append(ac.MRR(20))
            metric.append(ac.MRR(10))
            metric.append(ac.MRR(3))
            metric.append(cov.Coverage(20))
            metric.append(pop.Popularity(20))
    
            # create a dict of (textual algorithm description => class) to be evaluated
            algs = {}
    
            key = 'gru4rec2-' + loss + '-lr' + str(lr) + '-do' + str(drop) + '-mom' + str(momentum) + 't' + str(test_num)
            print('TESTING: ' + key)
            drop=float(drop)

            gru = gru4rec2.GRU4Rec(loss=loss, final_act='linear', hidden_act='tanh', layers=[100], batch_size=64, dropout_p_hidden=drop, learning_rate=lr, momentum=momentum, n_epochs=10,  n_sample=2048, sample_alpha=0, time_sort=True)
            algs[key] = gru
            
            '''
            Execution
            '''
            # load data
            
            print('data_path: ', data_path)
            
            train, test = loader.load_data(data_path, file_prefix, rows_train=limit_train, rows_test=limit_test, density=density_value, slice_num=slice)

            item_ids = train.ItemId.unique()
            
            # init metrics
            for m in metric:
                m.init(train)
            
            # train algorithms
            for k, a in algs.items():
                ts = time.time()
                print('fit ', k)
                a.fit(train)
                print(k, ' time: ', (time.time() - ts))
             
            # result dict
            res = {};
            
            # evaluation
            for k, a in algs.items():
                try:
                    res[k] = eval.evaluate_sessions(a, metric, test, train)
                    a.clear()
                except Exception:
                    a.clear()
                    del a
                    del algs
                    del metric
                    del train
                    del test
                    del gru
                    print('cleared gru with nan error')
                    raise
            
            # print results
            for k, l in res.items():
                for e in l:
                    print(k, ':', e[0], ' ', e[1])
             
            if export_csv is not None:
                
                fileH = open(export_csv, 'w+')
                fileH.write('Metrics;')
                
                for k, l in res.items():
                    for e in l:
                        fileH.write(e[0])
                        fileH.write(';')
                    break
                        
                fileH.write('\n')   
                    
                for k, l in res.items():
                    fileH.write(k)
                    fileH.write(';')
                    for e in l:
                        fileH.write(str(e[1]))
                        fileH.write(';')
                    fileH.write('\n')
                fileH.close()
                    
            if res[key][0][1] > best:  # new best found
                best = res[key][0][1]
                best_key = key
                
            print('CURRENT BEST: ' + best_key)
            print('WITH HR@20: ' + str(best))
            
            del drop
            del momentum
            del loss
            del lr
            del algs
            del metric
            del train
            del test
            del gru
            del res
            del item_ids
            del fileH

            for i in range(5): gc.collect()
            
            
        except Exception:
            
            traceback.print_exc()
            