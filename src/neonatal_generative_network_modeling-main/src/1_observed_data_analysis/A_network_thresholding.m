%% Network Thresholding %% FROM HERE ON I REPLACED THIS CODE WITH PYTHON. 
%{ 

Written by Alexa Mousley, MRC Cognition and Brain Sciences Unit
Email: alexa.mousley@mrc-cbu.cam.ac.uk

This script takes raw networks, performs consensus thresholding followed by 
absolute thresholding and binarization. With thresholds as is (consesus =
0.6 and binarisation = 325), the resulting 'binarized_connectomes' variable
should be exactly equal to that of the 'binarized_connectomes' variable 
published.

Note: If you are using the example data, the consensus thresholding will 
need to be much lower that 60% as the data is random and therefore not many 
'connections' will be in common across the fake participants

%}
%% Add paths and load data
% clear;clc;
% load("/Users/adrian/Downloads/neonatal_generative_network_modeling-main/data/raw/unthresholded_connectomes.mat") % (630, 90, 90)

%%/ Users/adrian/Documents/01_projects/14_4D_lab/src/neonatal_generative_network_modeling-main/src/set_paths.m
base_path = "/Users/adrian/Documents/01_projects/14_4D_lab";

% Add paths
% run('/Users/adrian/Downloads/neonatal_generative_network_modeling-main/set_paths.m');  % <<<<< Add path to set_paths file
run(base_path + '/src/neonatal_generative_network_modeling-main/src/set_paths.m')
% Load unthresholded networks
unthresholded_connectomes = load(base_path + "/data/preprocessed/00_just_converted_for_matlab_and_python/data_700mb_individual_connectomes/SC_68.mat").M; 
% unthresholded_connectomes = permute(unthresholded_connectomes, [3 2 1]);
%%
% dist = readtable('/Users/adrian/Documents/01_projects/14_4D_lab/neonatal_generative_network_modeling-main/data/preprocessed/consensus_10mb/02_fiber_length_mat_68.csv');
dist = readtable(base_path + "/data/preprocessed/00_just_converted_for_matlab_and_python/data_10_consensus/02_fiber_length_mat_68.csv");
% Output for this data: 

%%
nsub = size(unthresholded_connectomes,1); % Set number of participants

%% Perform consensus thresholding
% Set threshold
set = 0.6;                     % Percentage threshold
threshold = floor(nsub * set); % Calculate the threshold value based on the number of participants

k = unthresholded_connectomes ~= 0;     % Find nonzero elements (essentially creating a mask)
u = squeeze(sum(k, 1));        % Sum the values along the first dimension, squeezing the result to remove singleton dimensions

% Keep/Remove indices
ind = u < threshold;           % Find indices where the sum is less than the threshold
indkeep = u > threshold;       % Find indices where the sum is greater than the threshold

% Apply Threshold
for sub = 1:nsub
    A = squeeze(unthresholded_connectomes(sub,:,:)); % Select one participant
    A(ind)=0;                               % Remove edges with index
    consensus_thresholded(sub,:,:)=A;       % Save network 
end

% Look at the number of connections before and after thresholding
for sub = 1:nsub
    % Get the original network (A) and thresholded network (B)
    A = squeeze(unthresholded_connectomes(sub,:,:));
    B = squeeze(consensus_thresholded(sub,:,:));
    
    % Calculate the number of connections before and after thresholding
    before_threshold(sub) = nnz(A)/2;    % Count the number of nonzero elements in A and divide by 2 (due to symmetric connections)
    after_threshold(sub) = nnz(B)/2;     % Repeat for B
end

%% Perform absolute thresholding and binarize networks

% Set threshold
thr = 325;

% Initialize variables
binarized_connectomes = [];    % Empty array to store binarised connectomes
density = [];                  % Empty array to store density values

% Apply thresholding
for sub = 1:nsub
    W = squeeze(consensus_thresholded(sub,:,:));       % Extract the consensus thresholded connectome for the current participant
    weighted_connectome = threshold_absolute(W, thr);  % Apply the absolute threshold to the connectome
    density(sub) = density_und(weighted_connectome);   % Calculate the density of the weighted network
    B = zeros(size(weighted_connectome));              % Create a matrix of zeros with the same size as the weighted network
    B(find(weighted_connectome)) = 1;                  % Find the connections above the threshold and set them to 1
    binarized_connectomes(sub,:,:) = B;                % Store the binarized connectome 
end

% Print results
disp(sprintf('At this threshold, a mean density of %g%% is produced across the sample.',100*mean(density)));

%% Perform density-controlled thresholding for the density-controlled analysis
% This one takes long-ish

thr  = [1:4000];     % Create various thresholds to iterate through
nthr = length(thr);  % Number of thresholds to try

% Initialize variables
binarized_connectomes = [];    % Empty array to store binarised connectomes
density = [];                  % Empty array to store density values

% Iterate through participants
for sub = 1:nsub 
    % Loop through each threshold option
    for t = 1:nthr 
        over = 0;                                            % Create variable for if density is over-shot (<10%)
        W     = squeeze(unthresholded_connectomes(sub,:,:)); % Extract the connectome for the current participant
        weighted_connectome = threshold_absolute(W, thr(t)); % Apply the absolute threshold to the connectome
        d = density_und(weighted_connectome);                % Calculate density
        if round(d,3) == 0.100 || over == 1                  % If the density is equal to 10% or density has been over-shot (indicated by over == 1)
            B = zeros(size(weighted_connectome));            % Create a matrix of zeros with the same size as the weighted network
            B(find(weighted_connectome)) = 1;                % Find the connections above the threshold and set them to 1
            binarized_connectomes(sub,:,:) = B;              % Store the binarized connectome 
            density(sub) = d;                                % Save density
            %p_thr = [p_thr thr(t)];
            %p_d = [p_d d];
            break                                            % Break loop since 10% has been reched
        end
        if round(d,3) < 0.100                                % If density has surpassed 10% (i.e., density is < 10%)
            over = 1;                                        % Indicate that density has been over shot by changing 'over' variable                                 
            t = t-1;                                         % Go back one threshold and re-run (in order to save connectome that has not over-shot 10%)
        end 
    end
end

% Print results
disp(sprintf('At this threshold, a mean density of %g%% is produced across the sample.',100*mean(density)));



%% Alternative code

% 
% 
% % Inputs:
% %   unthresholded_connectomes: [nsub x n x n]
% %   thr: vector of absolute thresholds, sorted ascending
% %   target = 0.10
% 
% target = 0.10;
% nsub   = size(unthresholded_connectomes,1);
% n      = size(unthresholded_connectomes,2);
% 
% binarized_connectomes = zeros(nsub, n, n);
% density               = zeros(nsub,1);
% thr_used              = zeros(nsub,1);
% 
% for sub = 1:nsub
%     W0 = squeeze(unthresholded_connectomes(sub,:,:));
% 
%     % Quick edge-case checks
%     d_lo = density_und(threshold_absolute(W0, thr(1)));
%     d_hi = density_und(threshold_absolute(W0, thr(end)));
% 
%     if d_lo < target
%         % Even the loosest threshold is already below target → take thr(1)
%         idx = 1; d_sel = d_lo;
%     elseif d_hi >= target
%         % Even the strictest threshold is still ≥ target → take thr(end)
%         idx = numel(thr); d_sel = d_hi;
%     else
%         % Binary search for highest idx with density >= target
%         lo = 1; hi = numel(thr);
%         idx = 1; d_sel = d_lo;  % best-so-far (guaranteed to be >= target by guard above)
% 
%         while lo <= hi
%             mid = floor((lo + hi)/2);
%             Wmid = threshold_absolute(W0, thr(mid));
%             dmid = density_und(Wmid);
% 
%             % Round to 3 decimals to mirror your original stopping rule
%             if round(dmid,3) == round(target,3)
%                 idx = mid; d_sel = dmid;
%                 break
%             end
% 
%             if dmid >= target % originally >=
%                 % Still at/above target → move right to try a higher threshold
%                 idx  = mid;         % best-so-far
%                 d_sel = dmid;
%                 lo   = mid + 1;
%             else
%                 % Overshot (below target) → move left
%                 hi = mid - 1;
%             end
%         end
%     end
% 
%     % Apply selected threshold and binarize
%     % W_sel = threshold_absolute(W0, thr(idx)); -> original
%     W_sel = threshold_absolute(W0, thr(idx)+1); % -> changed
%     % B     = double(W_sel > 0);         % faster than find(); excludes diagonal if W0 had zeros there
%     B(find(weighted_connectome)) = 1; 
% 
%     binarized_connectomes(sub,:,:) = B;
%     density(sub)                   = d_sel;
%     thr_used(sub)                  = thr(idx);
% end
% 
% % Print results
% disp(sprintf('At this threshold, a mean density of %g%% is produced across the sample.',100*mean(density)));





%% Save binarized_connectomes 
save(processed_data_path + "/binarized_connectomes.mat")