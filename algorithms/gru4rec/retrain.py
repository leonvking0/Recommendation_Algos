# -*- coding: utf-8 -*-
"""
@author: mludewig
"""
from algorithms.gru4rec import gru4rec
import numpy as np

class GRU4Retrain:
    '''
    GRU4Retrain(layers, n_epochs=10, batch_size=50, dropout_p_hidden=0.5, learning_rate=0.05, momentum=0.0, adapt='adagrad', decay=0.9, grad_cap=0, sigma=0, init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False, lmbd=0.0, session_key='SessionId', item_key='ItemId', time_key='Time', n_sample=0, sample_alpha=0.75)
    
    Adapter for Gru4Rec that retrains the algorithm on the last N days in the end to shift the focus on recent trends. 

    Parameters
    -----------
    layers : 1D array
        list of the number of GRU units in the layers (default: [100] --> 100 units in one layer)
    n_epochs : int
        number of training epochs (default: 10)
    batch_size : int
        size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 50)
    dropout_p_hidden : float
        probability of dropout of hidden units (default: 0.5)
    learning_rate : float
        learning rate (default: 0.01)
    momentum : float
        if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the appropriate learning rate adaptation strategy, use None for standard SGD (default: 'adagrad')
    decay : float
        decay parameter for RMSProp, has no effect in other modes (default: 0.9)
    grad_cap : float
        clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
    init_as_normal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    reset_after_session : boolean
        whether the hidden state is set to zero after a session finished (default: True)
    loss : 'top1', 'bpr' or 'cross-entropy'
        selects the loss function (default: 'top1')
    hidden_act : 'tanh' or 'relu'
        selects the activation function on the hidden states (default: 'tanh')
    final_act : None, 'linear', 'relu' or 'tanh'
        selects the activation function of the final layer where appropriate, None means default (tanh if the loss is brp or top1; softmax for cross-entropy),
        cross-entropy is only affeted by 'tanh' where the softmax layers is preceeded by a tanh nonlinearity (default: None)
    train_random_order : boolean
        whether to randomize the order of sessions in each epoch (default: False)
    lmbd : float
        coefficient of the L2 regularization (default: 0.0)
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    n_sample : int
        number of additional negative samples to be used (besides the other examples of the minibatch) (default: 0)
    sample_alpha : float
        the probability of an item used as an additional negative sample is supp^sample_alpha (default: 0.75)
        (e.g.: sample_alpha=1 --> popularity based sampling; sample_alpha=0 --> uniform sampling)

    '''
    def __init__(self, layers, n_epochs=10, batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, adapt='adagrad', decay=0.9, grad_cap=0, sigma=0,
                 init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False, lmbd=0.0,
                 session_key='SessionId', item_key='ItemId', time_key='Time', n_sample=0, sample_alpha=0.75):
        self.gru = gru4rec.GRU4Rec( layers, n_epochs, batch_size, dropout_p_hidden, learning_rate, momentum, adapt, decay, grad_cap, sigma,
                 init_as_normal, reset_after_session, loss, hidden_act, final_act, train_random_order, lmbd,
                 session_key, item_key, time_key, n_sample, sample_alpha )
        
    
    def fit(self, data):
        
        self.gru.fit( data )
        
        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        days = 2
        sessions_retrain = session_max_times[session_max_times >= tmax-(86400*days)].index
        tmp = data[np.in1d(data.SessionId, sessions_retrain)]
        
        self.gru.fit( tmp, True )
            
    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        return self.gru.predict_next_batch(session_ids, input_item_ids, predict_for_item_ids, batch)
        
    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        return self.gru.predict_next(session_id, input_item_id, predict_for_item_ids, skip)
