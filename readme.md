Read Me
======================
Requirements
----------------------
* Anaconda 4.X (Python 3.5+)
* Pympler
* NumPy
* SciPy
* BLAS
* Pandas
* Theano
* Tensorflow
* Suitable datasets (https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0)

Installation (tested on Debian 8)
----------------------
1. Download and install Anaconda (https://www.continuum.io/).
	* NumPy, SciPy, BLAS, Pandas should automatically be included.
2. Install build essentials
	* apt-get install build-essential
	* On Windows the installation of mingw with Anaconda should work.
		* conda install mingw
3. Install Theano, Tensorflow and additional packages
	* conda install theano tensorflow pympler

Usage
----------------------
### Preprocessing
1. Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder data/rsc15/raw
2. Open the script run_preprocessing*.py to configure the preprocessing method and parameters
	* run_preprocessing_rsc15.py is for the RecSys challenge dataset.
	* run_preprocessing_tmall.py is for the TMall logs.
	* run_preprocessing_retailrocket.py is for the Retailrocket competition dataset.
	* run_preprocessing_clef.py is for the Plista challenge dataset.
	* run_preprocessing_music.py is for all music datasets (configuration of the input and output path inside the file).
3. Run the script

### Running experiments
1. You must have run the preprocessing scripts previously
2. Open and edit one of the run_test*.py scripts
	* run_test.py evaluates predictions for single split in terms of just the next item (HR@X and MRR@X)
	* run_test_pr.py evaluates predictions for single split in terms of all remaining items in the session (P@X, R@X, and MAP@X)
	* run_test_window.py evaluates predictions for sliding window split in terms of the next item (HR@X and MRR@X)
 	* run_test_buys.py evaluates buy events in the sessions (only for the rsc15 dataset).The script run_preprocessing.py must have been executed with method "buys" before.
	* The usage of all algorithms is exemplarily shown in the script. 
3. Run the script
4. Results and run times will be displayed and saved to the results folder as configured
