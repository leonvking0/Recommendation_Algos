import preprocessing.preprocess_music as pp
import time

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a sliding window approach  
'''
METHOD = "slice"

'''
data config (all methods) // change dataset here
'''
PATH = 'data/nowplaying/raw/' #30music,nowplaying,aotm,8tracks
PATH_PROCESSED = 'data/nowplaying/slices/' #30music,nowplaying,aotm,8tracks
FILE = 'nowplaying' #30music-200ks,nowplaying,playlists-aotm,8tracks

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 5
MIN_ITEM_SUPPORT = 2

'''
days test default config
'''
DAYS_FOR_TEST = 4

'''
slicing default config
'''
NUM_SLICES = 5 #offset in days from the first date in the data set
DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
DAYS_SHIFT = 60
#each slice consists of...
DAYS_TRAIN = 90
DAYS_TEST = 5

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    print( "START preprocessing ", METHOD )
    sc, st = time.clock(), time.time()
    
    if METHOD == "info":
        pp.preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
    
    elif METHOD == "org":
        pp.preprocess_org( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
        
    elif METHOD == "days_test":
        pp.preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_FOR_TEST )
    
    elif METHOD == "slice":
        pp.preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preproccessing ", (time.clock() - sc), "c ", (time.time() - st), "s" )
    