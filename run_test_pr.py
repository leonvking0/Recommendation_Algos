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

from evaluation import evaluation_multiple as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac
from evaluation.metrics import artist_coherence as coh
from evaluation.metrics import artist_diversity as div
from evaluation.metrics import coverage as cov
from evaluation.metrics import popularity as pop

import theano.misc.pkl_utils as pkl


if __name__ == '__main__':
    
    '''
    Configuration
    '''
    data_path = 'data/rsc15/single/'
    file_prefix = 'rsc15-clicks'

    limit_train = None #limit in number of rows or None
    limit_test = None #limit in number of rows or None
    density_value = 1 #randomly filter out events (0.0-1.0, 1:keep all)
    
    export_csv = None;
    export_csv = 'results/test-results-pr.csv'
    
    print( data_path )
     
    # create a list of metric classes to be evaluated
    metric = []
    metric.append( ac.Precision(20) )
    metric.append( ac.Precision(10) )
    metric.append( ac.Precision(5) )
    metric.append( ac.Precision(3) )
    metric.append( ac.Recall(20) )
    metric.append( ac.Recall(10) )
    metric.append( ac.Recall(5) )
    metric.append( ac.Recall(3) )
    metric.append( ac.MAP(20) )
    metric.append( ac.MAP(10) )
    metric.append( ac.MAP(5) )
    metric.append( ac.MAP(3) )
    metric.append( cov.Coverage(20) )
    metric.append( cov.Coverage(10) )
    metric.append( cov.Coverage(5) )
    metric.append( cov.Coverage(3) )
    metric.append( pop.Popularity(20) )
    metric.append( pop.Popularity(10) )
    metric.append( pop.Popularity(5) )
    metric.append( pop.Popularity(3) )
#     metric.append( div.ArtistDiversity(20) )
#     metric.append( coh.ArtistCoherence(20) )

    # create a dict of (textual algorithm description => class) to be evaluated
    algs = {}
    
    #baselines
    
    mk = markov.MarkovModel()
    algs['markov'] = mk
          
    sra = sr.SequentialRules( 10, weighting='div', pruning=20, last_n_days=None )
    algs['sr10-div'] = sra
       
    ara = ar.AssosiationRules();
    algs['ar'] = ara
      
    #knn
      
    iknn = iknn.ItemKNN()
    algs['iknn'] = iknn
      
    sknn = sknn.ContextKNN( 100, 500, similarity="cosine", extend=False )
    algs['sknn-100-500-cosine'] = sknn
       
    vmsknn = vsknn.VMContextKNN( 100, 2000, similarity="cosine", last_n_days=None, extend=False )
    algs['vsknn-200-2000-cosine'] = vmsknn
      
    ssknn = ssknn.SeqContextKNN( 100, 500, similarity="cosine", extend=False )
    algs['ssknn-100-500-cosine-div'] = ssknn
      
    sfsknn = sfsknn.SeqFilterContextKNN( 100, 500, similarity="cosine", extend=False )
    algs['sfsknn-100-500-cosine-div'] = ssknn
         
    #gr4rec2
     
    gru = gru4rec2.GRU4Rec(n_epochs=10, loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
    algs['gru-100-bpr-max-0.5'] = gru
     
    #session mf
     
    smf = smf.SessionMF( factors=100, batch=32, learn='adagrad_sub', learning_rate=0.085, momentum=0.2, regularization=0.005, dropout=0.3, skip=0.0, epochs=10, shuffle=-1, activation='linear', objective='top1_max' )
    algs['smf-top1-max'] = smf
     
    #filemodel to use pretrained pickled classes
#     fmdl = fm.FileModel( 'mdl/test-gru.mdl' )
#     algs['fm-gru'] = fmdl;
     
    #sequential recommendations adapter ()
     
    adpt = ad.Adapter(algo='bprmf')
    algs['bprmf'] = adpt
      
    adpt = ad.Adapter(algo='fism')
    algs['fism'] = adpt
      
    adpt = ad.Adapter(algo='fossil')
    algs['fossil'] = adpt
      
    adpt = ad.Adapter(algo='fpmc')
    algs['fpmc'] = adpt
     
    #weighted example
 
    hybrid = wh.WeightedHybrid( [vmsknn, sra], [0.5,0.5], fit=False )
    algs['whybrid-test-50-50'] = hybrid;
      
    hybrid = wh.WeightedHybrid( [vsknn.VMContextKNN( 100, 2000 ), sr.SequentialRules()], [0.5,0.5], fit=True )
    algs['whybrid-test-50-50-fit'] = hybrid;
    
    
    '''
    Execution
    '''
    #load data
    train, test = loader.load_data( data_path, file_prefix, rows_train=limit_train, rows_test=limit_test, density=density_value )
    item_ids = train.ItemId.unique()
    
    #init metrics
    for m in metric:
        m.init( train )
    
    #train algorithms
    for k,a in algs.items():
        ts = time.time()
        print( 'fit ', k)
        a.fit( train )
        print( k, ' time: ', ( time.time() - ts) )
        #print( k, ' memory: ', asizeof.asizeof(a) )
     
    #result dict
    res = {};
    
    #evaluation
    for k, a in algs.items():
        res[k] = eval.evaluate_sessions( a, metric, test, train )
    
    #print results
    for k, l in res.items():
        for e in l:
            print( k, ':', e[0], ' ', e[1] )
     
     
    if export_csv is not None:
        
        file = open(export_csv,'w+')
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
              