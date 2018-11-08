"""
Created on Fri Jun 26 17:27:26 2015

@author: ludewig
"""

import time
import os.path
import numpy as np
import pandas as pd
from _datetime import timezone, datetime


def load_data( path, file, rows_train=None, rows_test=None, slice_num=None, density=1, train_eval=False ):
    '''
    Loads a tuple of training and test set with the given parameters. 

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file. 
        This option will automatically filter the test set to only retain items included in the training set.  
    rows_test : int or None
        Number of rows to load from the test set file. 
    slice_num : 
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1). 
        The result is cached for the execution of multiple experiments. 
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)
    
    '''
    
    print('START load data') 
    st = time.time()
    sc = time.clock()
    
    split = ''
    if( slice_num != None and isinstance(slice_num, int ) ):
        split = '.'+str(slice_num)
    
    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'
    
    density_appendix = ''
    if( density < 1 ): #create sample
        
        if not os.path.isfile( path + file + train_appendix + split + '.txt.'+str( density ) ) :
            
            train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId':np.int64})
            test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId':np.int64} )
            
            sessions = train.SessionId.unique() 
            drop_n = round( len(sessions) - (len(sessions) * density) )
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            train = train[ ~train.SessionId.isin( drop_sessions ) ]
            train.to_csv( path + file + train_appendix +split+'.txt.'+str(density), sep='\t', index=False )
            
            sessions = test.SessionId.unique() 
            drop_n = round( len(sessions) - (len(sessions) * density) )
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            test = test[ ~test.SessionId.isin( drop_sessions ) ]
            test = test[np.in1d(test.ItemId, train.ItemId)]
            test.to_csv( path + file + test_appendix +split+'.txt.'+str(density), sep='\t', index=False )
    
        density_appendix = '.'+str(density)
            
    if( rows_train == None ):
        train = pd.read_csv(path + file + train_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64})
    else:
        train = pd.read_csv(path + file + train_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64}, nrows=rows_train)
        session_lengths = train.groupby('SessionId').size()
        train = train[np.in1d(train.SessionId, session_lengths[ session_lengths>1 ].index)]     
    
    if( rows_test == None ):
        test = pd.read_csv(path + file + test_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64} )
    else :
        test = pd.read_csv(path + file + test_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64}, nrows=rows_test )
        session_lengths = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, session_lengths[ session_lengths>1 ].index)]
    
#     rows_train = 10000
#     train = train.tail(10000)
        
    if( rows_train != None ):
        test = test[np.in1d(test.ItemId, train.ItemId)]
        session_lengths = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, session_lengths[ session_lengths>1 ].index)]
          
    #output
    data_start = datetime.fromtimestamp( train.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( train.Time.max(), timezone.utc )
    
    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    data_start = datetime.fromtimestamp( test.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( test.Time.max(), timezone.utc )
    
    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    print( 'END load data ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    
    return (train, test)


def load_buys( path, file ):
    '''
    Load all buy events from the youchoose file, retains events fitting in the given test set and merges both data sets into one

    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
        
    Returns
    --------
    out : pandas.DataFrame
        test with buys
    
    '''
    
    print('START load buys') 
    st = time.time()
    sc = time.clock()
        
    #load csv
    buys = pd.read_csv(path + file + '.txt', sep='\t', dtype={'ItemId':np.int64})
        
    print( 'END load buys ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    
    return buys
