import time

from pympler import asizeof

from algorithms.baselines import ar
from algorithms.baselines import sr
from algorithms.baselines import markov

from algorithms.knn import iknn
from algorithms.knn import cknn as sknn
from algorithms.knn import sfcknn as sfsknn
from algorithms.knn import scknn as ssknn
from algorithms.knn import vmknn as vsknn

from algorithms.gru4rec import gru4rec2
from algorithms.smf import smf
from algorithms.sbr_adapter import adapter as ad

from algorithms.filemodel import filemodel as fm
from algorithms.hybrid import weighted as wh

#from evaluation import evaluation as eval
from evaluation import evaluation as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac
from evaluation.metrics import artist_coherence as coh
from evaluation.metrics import artist_diversity as div
from evaluation.metrics import coverage as cov
from evaluation.metrics import popularity as pop
import theano.misc.pkl_utils as pkl 
import numpy as np
from _datetime import timezone
from _datetime import datetime
import gc


if __name__ == '__main__':
    
    DAYS = [60,30,20,15,10,5,3,2,1]
    
    '''
    Configuration
    '''
    data_path = 'data/rsc15/single/'
    file_prefix = 'rsc15-clicks'

    limit_train = None #limit in number of rows or None
    limit_test = None #limit in number of rows or None
    density_value = 1 #randomly filter out events (0.0-1.0, 1:keep all)
    
#     export_csv = None;
    export_csv = 'results/test-results-sparsity'
    
    print( data_path )
     
    # create a list of metric classes to be evaluated
    metric = []
    metric.append( ac.HitRate(20) )
    metric.append( ac.HitRate(10) )
    metric.append( ac.HitRate(5) )
    metric.append( ac.HitRate(3) )
    metric.append( ac.MRR(20) )
    metric.append( ac.MRR(10) )
    metric.append( ac.MRR(5) )
    metric.append( ac.MRR(3) )
#     metric.append( ac.MAP(20) )
#     metric.append( ac.MAP(10) )
    metric.append( cov.Coverage(20) )
#     metric.append( cov.Coverage(20) )
    metric.append( cov.Coverage(10) )
    metric.append( cov.Coverage(5) )
    metric.append( cov.Coverage(3) )
    metric.append( pop.Popularity(20) )
    metric.append( pop.Popularity(10) )
    metric.append( pop.Popularity(5) )
    metric.append( pop.Popularity(3) )
#     metric.append( div.ArtistDiversity(20) )
#     metric.append( coh.ArtistCoherence(20) )

    def get_algs():
        # create a dict of (textual algorithm description => class) to be evaluated
        algs = {}
        
        ara = ar.AssosiationRules()
        algs['ar'] = ara
        del ara
         
        sra = sr.SequentialRules( 10, weighting='div', pruning=20 )
        algs['sr10-div'] = sra
        del sra
        
        sknna = sknn.ContextKNN( 500, 1000, similarity="cosine" )
        algs['sknn-500-1000-cosine'] = sknna
        del sknna
        
        ssknna = ssknn.SeqContextKNN(100, 500, similarity="cosine")
        algs['ssknn-100-500-cosine'] = ssknna
        del ssknna
        
        knn = sfsknn.SeqFilterContextKNN(100, 500, similarity="cosine")
        algs['sfknn-100-500-cosine'] = knn
        del knn
        
        vsknna = vsknn.VMContextKNN(100, 1000, similarity="cosine")
        algs['svmknn-100-1000-cosine'] = vsknna 
        
        vsknna = vsknn.VMContextKNN(100, 2000, similarity="cosine")
        algs['svmknn-100-2000-cosine'] = vsknna
        del vsknna
        
        grunew = gru4rec2.GRU4Rec(loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
        algs['gru-100-bpr-max-0.5'] = grunew
        del grunew
        
        return algs
    
    '''
    Execution
    '''
    #load data
    train, test = loader.load_data( data_path, file_prefix, rows_train=limit_train, rows_test=limit_test, density=density_value )
    item_ids = train.ItemId.unique()
        
    for days in DAYS:
        
        max_time = train.Time.max()
        seconds = days * 60 * 60 * 24
        
        min_new = max_time - seconds
        if min_new < train.Time.min():
            print( days, 'too much. insert full dataset here.' )
            pass
        else: 
            #filter
            session_max_times = train.groupby('SessionId').Time.max()
            session_keep = session_max_times[ session_max_times > min_new ].index
        
            train_sparse = train[ np.in1d(train.SessionId, session_keep) ]
            
            test_sparse = test[np.in1d(test.ItemId,train_sparse.ItemId.unique())]
            session_lengths = test_sparse.groupby('SessionId').size()
            test_sparse = test_sparse[np.in1d(test_sparse.SessionId, session_lengths[ session_lengths>= 2 ].index)]
            
            data_start = datetime.fromtimestamp( train_sparse.Time.min(), timezone.utc )
            data_end = datetime.fromtimestamp( train_sparse.Time.max(), timezone.utc )
    
            print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
            format( len(train_sparse), train_sparse.SessionId.nunique(), train_sparse.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
            
            print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test_sparse), test_sparse.SessionId.nunique(), test_sparse.ItemId.nunique()))
    
            #init metrics
            for m in metric:
                m.reset()
                m.init( train )
            
            algs = get_algs()
            
            #train algorithms
            for k,a in algs.items():
                ts = time.time()
                print( "    fit ", k ) 
                a.fit( train_sparse )
                print( k, ' time: ', ( time.time() - ts) )
                  
            #result dict
            res = {};
            
            #evaluation
            for k, a in algs.items():
                res[k] = eval.evaluate_sessions( a, metric, test_sparse, train_sparse )
            
            #print results
            for k, l in res.items():
                for e in l:
                    print( k, ':', e[0], ' ', e[1] )
                         
            if export_csv is not None:
                
                file = open(export_csv+'.days'+str(days)+'.csv','w+')
                file.write('Metrics;')
                
                for k, l in res.items():
                    for e in l:
                        file.write(e[0])
                        file.write(';')
                    break
                        
                file.write('\n')   
                    
                for k, l in res.items():
                    file.write(k)
                    file.write(';')
                    for e in l:
                        file.write(str(e[1]))
                        file.write(';')
                    file.write('\n')
            
            del test_sparse, train_sparse, res, algs
            
            gc.collect()
            
            