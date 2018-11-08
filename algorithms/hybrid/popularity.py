# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: ludewig
"""
import pandas as pd
from builtins import str

class StrategicHybrid: 
    '''
    StrategicHybrid(algorithms, thresholds, bins, fit, item_key)
    
    Use different algorithms depending on the average popularity of the items in the current session.
    
    Parameters
    --------
    algorithms : list
        List of algorithms to combine weighted.
    thresholds : float
        Threshold that indicates up to which popularity bin an algorithm should be applied.
    bin : float
        Number of popularity bins (0,bins-1). The higher, the more popular.
    item_key : float
        Proper list of weights. Must have the same length as algorithms. 
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?
    
    '''    
    
    def __init__(self, algorithms, thresholds, bins=10, fit=True, item_key='ItemId'):
        self.algorithms = algorithms
        self.thresholds = thresholds
        self.run_fit = fit
        self.bins = bins
        self.item_key = item_key

    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        if self.run_fit:
            for a in self.algorithms:
                a.fit( data )
        
        pop = pd.DataFrame()
        pop['ItemPop'] = data.groupby( self.item_key ).size()
        pop[self.item_key] = pop.index
        
        data = data.merge( pop, on=self.item_key, how='inner' )
        data['PopBin'] = pd.qcut(data.ItemPop, self.bins, duplicates='drop')
        
        self.binmap = data[[self.item_key,'PopBin']].drop_duplicates()
        self.binmap.index = self.binmap[self.item_key]
        self.binmap.PopBin = self.binmap.PopBin.cat.codes
        del pop
           
        self.session = -1
        self.session_items = []
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        if( self.session != session_id ): #new session  
            self.session = session_id
            self.session_items = list()
        
        self.session_items.append( input_item_id )
        
        pop_lvl = self.get_pop_level( self.session_items )
                    
        for i, threshold in reversed(list(enumerate(self.thresholds))):
            if pop_lvl >= threshold:
                print( threshold )
                print( pop_lvl )
                print( self.algorithms[i].__class__.__name__ )
                predictions = self.algorithms[i].predict_next( session_id, input_item_id, predict_for_item_ids, skip=skip, type=type, timestamp=timestamp )
                break
        return predictions
            
    def get_pop_level(self, items):
        
        level = 0
        for i in items:
            level += self.binmap.ix[i].PopBin
            
        return level / len(items)
        