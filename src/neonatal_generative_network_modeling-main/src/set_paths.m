%% Set paths to load published data into scripts
%{

Written by Alexa Mousley, MRC Cognition and Brain Sciences Unit
Email: alexa.mousley@mrc-cbu.cam.ac.uk

Edit this script to contain the paths to the data you would like to use to
run the scripts.

%}

%% Base path
% base_path = "/Users/adrian/Downloads/neonatal_generative_network_modeling-main";
% base_path = "/Users/adrian/Documents/01_projects/14_4D_lab/neonatal_generative_network_modeling-main"
base_path = "/Users/adrian/Documents/01_projects/14_4D_lab";

%% Function/Toolbox paths

% Add Brain Connectivity Toolbox
%{
All graphy theory measures were calculated using the Brain Connectivity Toolbox
(https://sites.google.com/site/bctnet/home?authuser=0)

Toolbox publication:
Rubinov, M., & Sporns, O. (2010). Complex network measures of brain 
connectivity: uses and interpretations. Neuroimage, 52(3), 1059-1069.
%}
% addpath(base_path+"/BCT/2019_03_03_BCT");  %% CHANGED NOW EVERYWHERE TO
% FOLDER IN DOCUMENTS!
addpath(base_path+"neonatal_generative_network_modeling-main/BCT/2019_03_03_BCT");

% Add path to consensus network function
%{
Consensus network function can be found here: 
https://www.brainnetworkslab.com/coderesources

The function is documented in this publication:
Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2019). 
Distance-dependent consensus thresholds for generating group-representative
structural brain networks. Network neuroscience, 3(2), 475-496.

%}
% addpath(base_path+"neonatal_generative_network_modeling-main/distanceDependent");     

%% Data paths

data_path = base_path + "/data"; 

raw_data_path = data_path + "/raw"; 
addpath(raw_data_path); 
processed_data_path = data_path + "/processed"; 
addpath(processed_data_path);
preprocessed_data_path = data_path + "/preprocessed"; 
addpath(preprocessed_data_path);
% make sure the target folder exists: Is already done in preprocessing
if ~exist(preprocessed_data_path,'dir'), mkdir(preprocessed_data_path); end


downloaded_data_path = raw_data_path + "/downloaded_data";
atlas_path = downloaded_data_path + "/atlas"; 
% addpath(downloaded_data);

% Add path to demographic data
% addpath(downloaded_data_path+"/demographics"); 
% Add path to all observed networks 
% addpath(downloaded_data+"/observed_networks"); 
% Add path to all simulated networks 
% addpath(downloaded_data+"/simulated_networks"); 

% Add path to atlas data 
addpath(atlas_path);     

% Add path to derived data (e.g., graph theory measures)
% addpath(downloaded_data+"/derived_data"); 

% % Add path to density-controlled data
% addpath(downloaded_data+"/derived_data/density-controlled_data"); 

% Paths to propensity-matched data
% addpath(downloaded_data+"/propensity_matched_analysis");              % Networks
% addpath(downloaded_data+"/propensity_matched_analysis/demographics"); % Demographics
% addpath(downloaded_data+"/propensity_matched_analysis/derived_data"); % Derived data 
