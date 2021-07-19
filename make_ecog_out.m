%% make out struct of ecog from the mat files

addpath(genpath('/home/lgwilliams/scripts/'));

%% paths
data_dir = '/userdata/lgwilliams/neuropixels/data/NP04/ecog/';
blocks = {'EC237_B1','EC237_B11','EC237_B20','EC237_B27'};
data_types = {'HilbAA_70to150_8band','RawHTK_1000'};

% load events
out = table2struct(readtable(fullfile(data_dir, 'EC237_evnt_CORRECTED.csv')));

counter = 1;

% loop through blocks
for block = blocks
    
    % load mat
    hg_save_fname = char(strcat(data_dir, block, 'HG.mat'));
    d_hg = load(hg_save_fname);
    d_hg = d_hg.d_hg;
    lfp_save_fname = char(strcat(data_dir, block, 'LFP.mat'));
    d_lfp = load(lfp_save_fname);  
    d_lfp = d_lfp.d_lfp;
    
    % specify block number
    block_n = split(block{1}, '_');
    block_n = block_n{2};
    
    % make events
    for ei = 1:length(out)
        if strcmp(out(ei).block, block_n)
            tmin = ceil((out(ei).CorrectStartTime - 0.5)*1000);
            tmax = ceil((out(ei).StopTime + 1.0)*1000);
            out(counter).resp_hg = d_hg(:, tmin:tmax);
            out(counter).resp_lfp = d_lfp(:, tmin:tmax);
            % update
            counter = counter + 1;
            disp(counter);            
        end
    end
end

% save it
save_fname = char(strcat(data_dir, 'EC237_ECOG_out.mat'));
save(save_fname, 'out', '-v7.3');