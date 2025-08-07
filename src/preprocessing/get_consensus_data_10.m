%% paths %% THIS ONE IS THE REAL ONE! 
%% This file creates the folder "data/preprocessed/00_just_converted_for_python", and - as the name says, just turns the raw data into a format that allows it to be used with python. No thresholding, no binarization, nothing. 

base_path = "/Users/adrian/Documents/01_projects/14_4D_lab"; 
% raw_data_path = "/Users/adrian/Documents/01_projects/12_ma_thesis/data/raw"; 
raw_data_path = base_path + "/data/raw"; 
preprocessed_data_path = base_path + "/data/preprocessed/00_just_converted_for_matlab_and_python";
% make sure the target folder exists
if ~exist(preprocessed_data_path,'dir'), mkdir(preprocessed_data_path); end

%% Data 10 mb
data_10 = load(raw_data_path + "/consensus_connectomes_from_70_young_adults_10mb.mat") 

%% Data 700 mb
% data_700 = load(raw_data_path + "/consensus_connectomes_from_70_young_adults_700mb.mat")

%% Extract the data 
matrix_cells   = data_10.LauConsensus.Matrices;      % change here 
preprocessed_data_path_subfolder = preprocessed_data_path + "/data_10_consensus";
% make sure the target folder exists
if ~exist(preprocessed_data_path_subfolder,'dir'), mkdir(preprocessed_data_path_subfolder); end
extract_data_and_save_to_csv(matrix_cells, preprocessed_data_path_subfolder);

function extract_data_and_save_to_csv(matrix_cells, preprocessed_data_path)

%% Extract

%% ------------------------------------------------------------------------
%  EXPORT ALL MATRICES / ROI-INFO TO INDIVIDUAL CSV FILES
%  ------------------------------------------------------------------------
%  Assumptions:
%  • The 5×5 cell array is in variable `MatCell`
%    (e.g.  MatCell = data_10.LauConsensus.Matrices;)
%  • Column layout           | Row layout
%    ---------------------------------------------------------
%    1 = 01_weighted_adj_mat | 1 = _68
%    2 = 02_fiber_length_mat | 2 = _114
%    3 = 03_fc_mat           | 3 = _219
%    4 = 04_coordinates      | 4 = _448
%    5 = { 05_1_roi_name     | 5 = _1000
%            05_2_short_name
%            05_3_rsn_name
%            05_4_hemisphere }
%  • Matlab R2019a or newer (uses writematrix / writecell)
%  ------------------------------------------------------------------------

% ---------- 1.  user-adjustable definitions ------------------------------
colNames  = ["01_weighted_adj_mat" , ...
    "02_fiber_length_mat" , ...
    "03_fc_mat"           , ...
    "04_coordinates"      , ...
    "05_1_roi_name"       , ...
    "05_2_short_name"     , ...
    "05_3_rsn_name"       , ...
    "05_4_hemisphere"];                 %#ok<CLARR>
rowNames  = ["_68","_114","_219","_448","_1000"]; %#ok<CLARR>
% -------------------------------------------------------------------------


% ---------- 2.  main export loop ----------------------------------------
for r = 1:size(matrix_cells,1)          % each resolution
    % --- columns 1–4 map 1:1 --------------------------------------------
    for c = 1:4
        thisData = matrix_cells{r,c};
        if isempty(thisData),  continue;  end     % skip blanks

        fileName = sprintf('%s%s.csv',colNames(c),rowNames(r));
        fullPath = fullfile(preprocessed_data_path,fileName);

        if iscell(thisData)
            writecell(thisData,fullPath);
        else
            writematrix(thisData,fullPath);
        end
    end

    % --- column 5 is itself a 1×4 cell-array ----------------------------
    roiBundle = matrix_cells{r,5};   % expect {roi_name  short  rsn  hemi}
    if isempty(roiBundle) || ~iscell(roiBundle),  continue;  end

    for k = 1:4
        thisData = roiBundle{1,k};
        fileName = sprintf('%s%s.csv',colNames(4+k),rowNames(r));
        fullPath = fullfile(preprocessed_data_path,fileName);

        if iscell(thisData)
            writecell(thisData,fullPath);
        else
            writematrix(thisData,fullPath);
        end
    end
end

fprintf('All exports finished.  Files are in:\n   %s\n',preprocessed_data_path);


end