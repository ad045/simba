%% Export data_700.connMatrices.SC  →  individual *.mat files. THIS ONE IS THE REAL ONE. 
%% This file creates the folder "data/preprocessed/00_just_converted_for_python", and - as the name says, just turns the raw data into a format that allows it to be used with python. No thresholding, no binarization, nothing. 
%paths 

base_path = "/Users/adrian/Documents/01_projects/14_4D_lab"; 
% raw_data_path = "/Users/adrian/Documents/01_projects/12_ma_thesis/data/raw"; 
raw_data_path = base_path + "/data/raw"; 
preprocessed_data_path = base_path + "/data/preprocessed/00_just_converted_for_matlab_and_python/data_700mb_individual_connectomes";
% make sure the target folder exists
if ~exist(preprocessed_data_path,'dir'), mkdir(preprocessed_data_path); end



%% Data 700 mb
data_700 = load(raw_data_path + "/consensus_connectomes_from_70_young_adults_700mb.mat"); 

SCcell  = data_700.connMatrices.SC;      % 5×1 cell array shown in the screenshot

for k = 1:numel(SCcell)
    M = SCcell{k};                % N×N×70 double
    if isempty(M),  continue, end % safety
    
    % --- switch dimensions: N×N×70  →  70×N×N
    M = permute(M,[3 1 2]);
    
    N        = size(M,2);                       % 68, 114, 219, ...
    fileName = sprintf('SC_%d.mat',N);          %  SC_68.mat, ...
    
    save(fullfile(preprocessed_data_path,fileName),'M','-v7.3') % -v7.3 supports >2 GB
    % 
    % fileName = sprintf('SC_%d.mat',N);          %  SC_68.mat, ...
    % writematrix(N, 'SC_%d.csv');

    

end

fprintf('Done.  Files are in: %s\n',preprocessed_data_path);

