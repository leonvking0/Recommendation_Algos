import pandas as pd
import numpy as np
from helper.node import PatternNode
import time

def get_stats( dataframe, name='test' ):
    print( 'get_stats ',name )
    
    res = {}
    
    res['STATS'] = ['STATS']
    res['name'] = [name]
    res['actions'] = [len(dataframe)]
    res['items'] = [ dataframe.ItemId.nunique() ]
    res['sessions'] = [ dataframe.SessionId.nunique() ]
    res['time_start'] = [ dataframe.Time.min() ]
    res['time_end'] = [ dataframe.Time.max() ]
    
    res['unique_per_session'] = dataframe.groupby('SessionId')['ItemId'].nunique().mean()
    
    res = pd.DataFrame(res)

    res['actions_per_session'] = res['actions'] / res['sessions']
    res['actions_per_items'] = res['actions'] / res['items']
    #res['sessions_per_action'] = res['sessions'] / res['actions']
    res['sessions_per_items'] = res['sessions'] / res['items']
    #res['items_per_actions'] = res['items'] / res['actions']
    res['items_per_session'] = res['items'] / res['sessions']
    res['span'] = res['time_end'] - res['time_start']
    
    return res

def sequential_indicators( train, name='test' ):
    
    print( 'sequential_indicators ',name )
    
    train['ItemIdNext'] = train['ItemId'].shift(-1).where(train['SessionId'].shift(-1) == train['SessionId'], np.nan)

    sequences = pd.DataFrame()
    sequences['count'] = train.dropna(axis=0, how='any').groupby( ['ItemId','ItemIdNext'] ).size()
    
    sums = pd.DataFrame()
    sums['size'] = sequences.groupby( ['count'] ).size()
    sums = sums.reset_index()
    sums['prod'] = sums['count'] * sums['size']
    
    sumf = sums['prod'].sum()
    
    res = {}
    res['name'] = [name]
    res['seq'] = [len( sequences )]
    res['seq_count_mean'] = [sequences['count'].mean()]
    res['seq_count_max'] = [sequences['count'].max()]
    res['seq_sum'] = [sumf]
    
    res = pd.DataFrame(res)
    
    res['seq_sum_tupel'] = res['seq_sum'] / res['seq']
    res['seq_sum_train'] = res['seq_sum'] / len( train )
    res['seq_sum_items'] = res['seq_sum'] / train.ItemId.nunique() 
    res['seq_sum_session'] = res['seq_sum'] / train.SessionId.nunique()
    
    res['seq_count_mean_tupel'] = res['seq_count_mean'] / res['seq']
    res['seq_count_mean_train'] = res['seq_count_mean'] / len( train )
    res['seq_count_mean_items'] = res['seq_count_mean'] / train.ItemId.nunique() 
    res['seq_count_mean_session'] = res['seq_count_mean'] / train.SessionId.nunique()
    
    res['seq_count_max_tupel'] = res['seq_count_max'] / res['seq']
    res['seq_count_max_train'] = res['seq_count_max'] / len( train )
    res['seq_count_max_items'] = res['seq_count_max'] / train.ItemId.nunique() 
    res['seq_count_max_session'] = res['seq_count_max'] / train.SessionId.nunique()
    
    return res

def tree_indicators(train, name='test'):
    
    print( 'tree_indicators ',name )
    print( 'max session:', ( train.groupby('SessionId').size().max() ) )
    
    train['session_size'] = train.groupby('SessionId')['SessionId'].transform('count')
    train = train[ train['session_size'] <= 250 ] 
    del train['session_size']
    
    index_session = train.columns.get_loc( 'SessionId' )
    index_item = train.columns.get_loc( 'ItemId' )
    
    curr_session = -1
    count = 0
    tree = PatternNode(-1)
    
    
    tstart = time.time()
    for row in train.itertuples( index=False ):
            
        session_id, item_id = row[index_session], row[index_item]
        
        if session_id != curr_session:
            curr_session = session_id
            last_items = []
            #print('new session')
        else: 
            combi = last_items + [item_id]
            for start in range(len(combi)-1):
                tree.add(combi[start:])
                
        last_items.append(item_id) 
    
        count += 1 
              
        if count % 1000000 == 0:
            print( 'finished row {} of {} in {}s'.format( count, len(train), ( time.time() - tstart ) ) )
    
    width_levels = 5
    prune_till = 10
           
    res = {}
    
    avg_width = {}
    for j in range(1,width_levels+1):
        avg_width[j] = 0
    
    avg_avg_depth = 0
    avg_max_depth = 0
    
    for i in range( 2, prune_till+1 ):
        print( 'tree features for min_count ', i )
        tree.prune(i)
                
        for j in range(1,width_levels+1):
            res['tree'+str(i)+'_width'+str(j)] = [ tree.width[j] if j in tree.width else 0 ]
            avg_width[j] += res['tree'+str(i)+'_width'+str(j)][0]
            
        res['tree'+str(i)+'_depth_max'] = [tree.depth]
        avg_max_depth += tree.depth
        
        dsum = 0
        for child in tree.children.items():
            dsum += child[1].depth
            
        res['tree'+str(i)+'_depth_avg'] = [ ( dsum / len(tree.children) ) if len(tree.children) > 0 else 0]
        avg_avg_depth += res['tree'+str(i)+'_depth_avg'][0]
    
    width_avg = 0
    for j in range(1,width_levels+1):
        res['tree_width'+str(j)+'_avg'] = [ avg_width[j] / 5 ]
        width_avg += res['tree_width'+str(j)+'_avg'][0]
    
    res['tree_width_avg'] = [ width_avg / width_levels ]
    res['tree_depth_avg'] = [ avg_avg_depth / (prune_till-1) ]
    res['tree_max_depth_avg'] = [ avg_max_depth / (prune_till-1) ]
    res['name'] = [name]
    
    res = pd.DataFrame(res)

    return res
    