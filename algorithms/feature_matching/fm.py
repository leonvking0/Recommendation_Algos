# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: ludewig
"""
import math
import time

import pandas as pd

class FeatureMatching:
    '''
    FeatureMatching(algorithm, features, store)
            
    Parameters
    --------
    algorithms : class
        Algorithm to produce ranking.
    features : list
        List of feature names to match
    store: string
        Path to the dressipi store to load a map with item information. 
    rerank: int
        -1: all
        n: number of top recommendations to rerank
    
    '''    
    
    def __init__(self, algorithm, features=['Cat','Style','Pattern','Fit','Length'], store='store.h5', rerank=-1 ):
        self.algorithm = algorithm
        self.features = features
        self.store = store
        self.rerank = rerank
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.session_profile = {}

    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        self.algorithm.fit( data )
        store = pd.HDFStore( self.store )
        
        self.items = self.load_item_data( store )
        
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view'):
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
            self.session_profile = {}
        
        if type == 'view':
            self.session_items.append( input_item_id )
            
            if input_item_id in self.items: #add features to session profile if existent
                for feature in self.features:
                    if feature in self.items[input_item_id]:
                        if feature not in self.session_profile:
                            self.session_profile[feature] = set()
                        self.session_profile[feature].add( self.items[input_item_id][feature] )
            
        
        predictions = self.algorithm.predict_next( session_id, input_item_id, predict_for_item_ids, skip, type )
       
        if skip:
            return
        
        ca, cb = 0, 0
        avg = 0.0
        
        if self.rerank > 0:
            iset = predictions.nlargest( self.rerank ).index
        else:
            iset = set( predictions.index )
        
        values = []
        for (item, score) in predictions.iteritems():
            if item in iset and score > 0:
                add = self.match(item, self.session_profile)
                avg += add
                values.append( score + add )
                cb += 1
            else:
                values.append(score)
                ca += 1
        
        predictions = pd.Series(data=values, index=predictions.index)
                      
        return predictions
    
    def match(self, item, profile):
        hit = 0
        for feature in self.features:
            if feature in profile and item in self.items and feature in self.items[item] and self.items[item][feature] in profile[feature]:
                hit += 1
                
        return hit
       
    def load_item_data(self, store):
    
        cats = store['categories']
        cat_names = store['category_names']
        cat_names = cat_names.set_index(['Cat'])
        features = store['features']
        feature_names = store['feature_names']
        feature_names = feature_names.set_index(['Feature'])
        
        items = {}
        
        for row in cats.itertuples(index=False):
            item, cat = row[0], row[1]
            if not item in items:
                items[item] = {}
            items[item]['Cat'] = cat
            items[item]['CatName'] = cat_names.ix[cat]['CatName']
            
            parent = cat_names.ix[ cat ][ 'CatParent' ]
            if not math.isnan( float(parent) ) :
                items[item]['ParentCat'] = parent
                items[item]['ParentCatName'] = cat_names.ix[ parent ][ 'CatName' ]
            
        brands = set()
        specbrand = set()
        styles = set()
        patterns = set()
            
        for row in features.itertuples(index=False):
            item, feature, content = row[0], row[2], row[3]
            if not item in items:
                items[item] = {}
                
            name = feature_names.ix[feature]['FeatureName']
            items[item][name] = content
            
            if row[2] == -1:
                brands.add(content)
            
            if name == 'Style':
                styles.add(content)
              
            if name == 'Brand Specific':
                specbrand.add(content) 
                
            if name == 'Pattern':
                patterns.add(content) 
        
        print( 'patterns: ',len(patterns))
        print( 'styles: ',len(styles))
        print( 'brands: ',len(brands))
          
        return items