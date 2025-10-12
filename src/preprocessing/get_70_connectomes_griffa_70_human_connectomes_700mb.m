%% Export data_700.connMatrices.SC  →  individual *.mat files.
%% This file creates the folder "data/preprocessed/00_just_converted_for_python", and - as the name says, just turns the raw data into a format that allows it to be used with python. No thresholding, no binarization, nothing. 

% Paths
base_path = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code"; 
raw_data_path = base_path + "/data/raw/griffa_70_human_connectomes_dataset"; 
preprocessed_data_path = base_path + "/data/preprocessed/griffa_70_human_connectomes_dataset/00_preprocessed";
% make sure the target folder exists
if ~exist(preprocessed_data_path,'dir'), mkdir(preprocessed_data_path); end

%% Data 700 mb
data_700 = load(raw_data_path + "/connectomes_from_70_young_adults_700mb.mat"); 

% Import data 
SCcell  = data_700.connMatrices.SC;      % 5×1 cell array shown in the screenshot

% Extract loop 
for k = 1:numel(SCcell)
    % get precise data 
    M = SCcell{k};                % Get data: N×N×70 double
    if isempty(M),  continue, end % safety
    M = permute(M,[3 1 2]);       % switch dimensions: N×N×70  →  70×N×N
    N = size(M,2);                % get size: 68, 114, 219, ...

    % save data 
    fileName = sprintf('SC_%d.mat',N);  % results in SC_68.mat, ...
    save(fullfile(preprocessed_data_path,fileName),'M','-v7.3') 
    
end

fprintf('Done.  Files are in: %s\n',preprocessed_data_path);

