# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: mludewig
"""

import numpy as np
import pandas as pd


class Remind:
    '''
    Remind()
    
    Popularity predictor that gives higher scores to items with larger support.
    
    The score is given by:
    
    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    num_days : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    time_key : string
        The header of the timestamp column in the training data. (Default value: 'Time')
    
    '''
    
    def __init__(self):
        return
            
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        self.session = -1
        self.session_items = []
        self.count = 0
        self.evals = 0
        
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False):
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
        
        self.session_items.append( input_item_id )
            
        if skip:
            return
        
        rank = 1
        preds = np.zeros(len(predict_for_item_ids))
        preds = pd.Series(data=preds, index=predict_for_item_ids)
        for item in reversed(self.session_items):
            preds[item] = 1.0 / rank;
            rank += 1
        
        return preds
        