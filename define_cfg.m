% configuration file for neuropixels preprocessing

% init struct
cfg = struct;

% init paths
base_dir = '/userdata/lgwilliams/neuropixels';
save_dir = char(strcat(base_dir, '/data/out_structs/'));

% user defined subj1
cfg(1).subject = 'NP04';
cfg(1).blocks = {'B2'};
cfg(1).crop_times = {[240, 840]};
cfg(1).chanMapFile = 'Intraop1_g0_t0.imec0.ap_kilosortChanMap.mat'; %'neuropixPhase3B2_kilosortChanMap.mat';
cfg(1).NchanTOT = 385;
cfg(1).ks_dir = char(strcat(base_dir, '/data/', cfg(1).subject, '/kilosort_out'));
cfg(1).ks_csv = 'agreed_upon_units-updated.csv';
cfg(1).ks_sorters = {'lg', 'mkl', 'ks'};

% user defined subj2...
cfg(2).subject = 'NP10';
cfg(2).blocks = {'B2', 'B3', 'B4'};
cfg(2).crop_times = {[390, 910], [230, 740], [385, 900]};
cfg(2).chanMapFile = 'Intraop1_g0_t0.imec0.ap_kilosortChanMap.mat'; %'neuropixPhase3B2_kilosortChanMap.mat';
cfg(2).NchanTOT = 385;

% NP12
cfg(3).subject = 'NP12';
cfg(3).blocks = {'tcat'};
cfg(3).crop_times = {[410, 1845]}; % last spike is 1430 (+ 410)
cfg(3).chanMapFile = 'Intraop1_g0_t0.imec0.ap_kilosortChanMap.mat'; %'neuropixPhase3B2_kilosortChanMap.mat';
cfg(3).NchanTOT = 385;
cfg(3).ks_dir = char(strcat(base_dir, '/data/', cfg(3).subject, '/kilosort_out'));
cfg(3).ks_csv = 'NP12_good_units.csv';
cfg(3).ks_sorters = {'lg'};

% NP13
cfg(4).subject = 'NP13';
cfg(4).blocks = {'B2'};
cfg(4).crop_times = {[540, 1080]};
cfg(4).chanMapFile = 'Intraop1_g0_t0.imec0.ap_kilosortChanMap.mat'; %'neuropixPhase3B2_kilosortChanMap.mat';
cfg(4).NchanTOT = 385;

% some of the fields are just automatically generated based off user input

% lengths
for cfg_n = 1:length(cfg)
    cfg(cfg_n).n_blocks = length(cfg(cfg_n).blocks);
    cfg(cfg_n).raw_folder = strcat(base_dir, '/data/', cfg(cfg_n).subject, '/raw/');
end

