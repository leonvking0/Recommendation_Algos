# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: mludewig
"""

from datetime import datetime as dt
from datetime import timedelta as td
import time

import numpy as np
import pandas as pd


class TreeSequentialRules: 
    '''
    TreeSequentialRules( extend_over_session=False, last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time' )
        
    Parameters
    --------
    extend_over_session : boolean
        Update the recommendation score over the session or create a new list per action. (Default value: False)
    last_n_days : int
        Only use the last N days of the data for the training process. (Default value: None)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_key : string
        The data frame key for the item identifier. (Default value: ItemId)
    time_key : string
        The data frame key for the timestamp. (Default value: Time)
    
    '''
    
    def __init__( self, extend_over_session=False, last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time' ):
        self.last_n_days = last_n_days
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.session = -1
        self.session_items = []
        self.extend_over_session = extend_over_session
            
    def fit( self, data ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        
            
        '''
        
        if self.last_n_days != None:
            
            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]
        
        else: 
            train = data
            
        cur_session = -1
        last_items = []
        tree = TreeSequentialRules.PatternNode(-1)
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        tstart = time.time()
        count = 0
        
        for row in train.itertuples( index=False ):
            
            session_id, item_id = row[index_session], row[index_item]
            
            if session_id != cur_session:
                cur_session = session_id
                last_items = []
                #print('new session')
            else: 
                combi = last_items + [item_id]
                for start in range(len(combi)-1):
                    tree.add(combi[start:])
                    
            last_items.append(item_id) 
        
            count += 1 
            
#             if count % 50 == 0:
#                 print( tree )
#                 exit()
                  
            if count % 1000000 == 0:
                print( 'finished row {} of {} in {}s'.format( count, len(train), ( time.time() - tstart ) ) )
        
        self.tree = tree
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.preds = np.zeros( len(predict_for_item_ids) )
            
        
        if type == 'view':
            self.session_items.append( input_item_id )
            
        if skip:
            return
        
        preds = np.zeros( len(predict_for_item_ids) )
        if self.extend_over_session:
            preds = self.preds
        
#         res = self.tree.get(self.session_items[-1:])
#         for e in res:
#             preds[ predict_for_item_ids == e[0] ] = e[1]
        
        for i in range( len( self.session_items ) ):
            res = self.tree.get(self.session_items[-(i+1):])
            for e in res:
                current = preds[ predict_for_item_ids == e[0] ]
                preds[ predict_for_item_ids == e[0] ] = current + e[1]
        
        series = pd.Series(data=preds, index=predict_for_item_ids)
        series = series / series.max()
        
        self.preds = preds
        
        return series
    
    
    class PatternNode:
        def __init__(self, id):
            self.children = {}
            self.key = id
            self.count = 0
        
        def add(self, seq):
            if len(seq) > 0:
                item = seq[0]
                if not item in self.children:
                    self.children[item] = TreeSequentialRules.PatternNode(item)
                if len( seq ) == 1:
                    self.children[item].count += 1
                self.children[item].add( seq[1:] )
                
        def get(self, seq):
            if len(seq) > 0:
                item = seq[0]
                if not item in self.children:
                    return []
                if item in self.children and len(seq) == 1:
                    return [ (item[0], item[1].count) for item in self.children[item].children.items() ]
                return self.children[item].get( seq[1:] )
            else:
                return []
            
        def __str__(self, level=0):
            ret = "\t"*level + repr(self.key) + '('+ str(self.count) +')' + "\n"
            for child in self.children.items():
                ret += child[1].__str__(level+1)
            return ret            
                
                        