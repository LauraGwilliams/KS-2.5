%% define paths

sys_flag = 'matlab'; % 'matlab', 'python' - which language will you use?

addpath(genpath('/home/matt_l/matlab/npy-matlab/'))
% addpath(genpath('/home/matt_l/matlab/KS_LG_MKL/')) % path to kilosort folder
addpath(genpath('/home/matt_l/matlab/KS-2.5'))
addpath(genpath('/home/matt_l/matlab/neuropixels/')) % path to kilosort folder
addpath(genpath('/home/matt_l/matlab/TIMIT')) % path to timit preproc

timitPath = '/home/matt_l/matlab/TIMIT';
rootZ = '/userdata/matt_l/neuropixels/NP04/raw/NP04_B2_g0/NP04_B2_g0_imec0/'; % the raw data binary file is in this folder
rootH = rootZ; % path to temporary binary file (same size as data, should be on fast SSD)
pathToYourConfigFile = '/home/matt_l/matlab/KS_LG_MKL/configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
chanMapFile = 'Intraop1_g0_t0.imec0.ap_kilosortChanMap.mat'; %'neuropixPhase3B2_kilosortChanMap.mat';

%% params
ops.trange    = [240 840]; % time range to sort
ops.NchanTOT  = 385; % total number of channels in your recording

% filtering
%ops.fslow = 10000;
%ops.fshigh = 300;

%% deriv paths
run(fullfile(pathToYourConfigFile, 'configFile384.m'))
ops.fproc   = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = fullfile(pathToYourConfigFile, chanMapFile);

%% load params

% load the channel map
load(ops.chanMap);

% load CORRECTED! event structure
switch sys_flag
    case 'matlab'
        % load event structure
        events = load(fullfile(rootZ, 'NP04_evnt_corrected.mat'));
        evnt = events.evnt;
        evnt_orig = load(fullfile(rootZ, 'NP04_evnt.mat'));
        evnt_orig = evnt_orig.evnt;
        for i = 1:length(evnt)
            tdiff(i) = evnt_orig(i).StartTime - evnt(i).StartTime;
        end
    case 'python'
        %events = load(fullfile(rootZ, 'NP04_evnt.mat'));
        %evnt = events.evnt;
        evnt = table2struct(readtable(fullfile(rootZ, 'NP04_evnt_CORRECTED.csv')));
end

% fix evnt paths
for i = 1:length(evnt)
    name = evnt(i).wname;
    tmp = strsplit(name,'@');
    evnt(i).wname = [timitPath '/@' tmp{2}];
end

% load the true channel ordering
ch_struct = load(fullfile(rootZ, 'NP04_chMap.mat'));
value_cut = ch_struct.chMap(find(connected == 0));

% which channels we use are defined in the physical space of the probe, not
% the chronological order of the data
bad_idx = find(ch_struct.chMap < 10 | ch_struct.chMap > 320);
connected(bad_idx) = 0;
connected(find(ch_struct.chMap == 186)) = 0;
connected(find(ch_struct.chMap == 187)) = 0;
ops.connected = connected;

%channel_reorderer = ch_struct.chMap(connected);  % use boolean to subset

% edit the 'connected' object to remove the bad channels from the
% preprocessing
%connected(1:10) = 0;
%connected(320:384) = 0;  % 342:384
%ops.connected = connected;

%% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s \n', rootZ)

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 5; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option.

% is there a channel map file in this folder?
fs = dir(fullfile(rootZ, 'chan*.mat'));
if ~isempty(fs)
    ops.chanMap = fullfile(rootZ, fs(1).name);
end

% find the binary file
fs = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*df.dat'))];
ops.fbinary = fullfile(rootZ, fs(1).name);

% preprocess data to create temp_wh.dat
sfreq = 30000;
decimate = 1;
new_sfreq = sfreq/decimate;

% change the NT to be an integer of the resulting sfreq
ops.NT = sfreq*2;

% get data
all_data_ds = preprocessData_returnMatrix(ops);

%% load anin
anin_path = '/userdata/matt_l/neuropixels/NP04/raw/NP04_B2_g0';
aninBin_fname = 'NP04_B2_g0_t0.nidq.bin';
aninMeta_fname = 'NP04_B2_g0_t0.nidq.meta';

% load it
aninmeta = ReadMeta(aninMeta_fname, anin_path);

% Get first one second of data
nSamp_anin = floor(1.0 * SampRate(aninmeta));

% timing
t0 = round(str2double(aninmeta.niSampRate)*240);
n_samples_in_seconds = 840-240;
t1 = round(str2double(aninmeta.niSampRate)*n_samples_in_seconds);
aninArray = ReadBin(t0,t1,aninmeta,aninBin_fname,anin_path);

%% OK, be careful in the channel re-ordering, please

% the chanMap is the full 1:384 channel list in chronological order, and connected is the boolean
% of which channels we actually used, and ch_struct.chMap are the
% physically ordered channels.

% 1) subset the physically ordered channels
% this tells us which of the original channels exist, in which order
phys_ch_subset = ch_struct.chMap(connected)';

% 2) subset the chron channels
chron_ch_subset = chanMap(connected);

% 3) get sorting indices of the physical - thats it!
[sorted_vals, sort_idx] = sort(phys_ch_subset);

% 4) these are the indices of the original space that we used
chs_original = chron_ch_subset(sort_idx);

%three_spaces = [phys_ch_subset, chron_ch_subset, chs_original, [1:310]'];
%disp(three_spaces);

%% load the spike sorted output from the .npy files
spike_times = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/spike_times.npy');
spike_clusters = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/spike_clusters.npy');
sua_label = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/sua_label.npy');
mua_label = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/mua_label.npy');

% derive params
n_spike_events = length(spike_times);
max_spike_time = max(spike_times);
n_clusters = length(unique(spike_clusters));
n_sua = length(sua_label);
n_mua = length(mua_label);

% because of the batching, we might end up with more time samples than we
% asked for. we will see later how we want to deal with this. but for now,
% lets just ignore any sample which is longer than 600 seconds.
desired_max_tsamp = sfreq * n_samples_in_seconds;

% 1) init the spike train matrix for SUA and MUA
spikes_sua = zeros(n_sua, desired_max_tsamp);
spikes_mua = zeros(n_mua, desired_max_tsamp);

% 2) loop through each spike event and put into the correct bin
for ev = 1:n_spike_events

    % get time and label
    s_time = spike_times(ev);
    s_label = spike_clusters(ev);

    % the batching issue - maybe lets fix this
    if s_time < desired_max_tsamp

    % get identity
    idx_sua = find(sua_label == s_label);
    idx_mua = find(mua_label == s_label);

    % put into correct place
    if sum(idx_sua) > 0
        spikes_sua(idx_sua, s_time) = 1;
    end
    if sum(idx_mua) > 0
        spikes_mua(idx_mua, s_time) = 1;
    end
    disp(s_time);

    end
end

%% make into a struct

new_fs = new_sfreq;

for j = 1:length(evnt)
    out(j).name = evnt(j).name;
end

for j = 1:length(evnt)
    out(j).name = evnt(j).name;
end

% params
tmin = 0.5;
tmax = 1.0;
nplot_samp = (tmax*new_sfreq) + (tmin*new_sfreq) + 1;
n_plot_audsamp = (tmax*nSamp_anin) + (tmin*nSamp_anin) + 1;

% evoked = zeros(length(evnt), size(all_data_ds,1), nplot_samp);
spike_ev = zeros(length(evnt), size(all_data_ds,1), 2*new_sfreq);
spike_ds = zeros(length(evnt), size(all_data_ds,1), 2000);
evoked_audio = zeros(length(evnt), n_plot_audsamp);
corrcoefs = zeros(1, length(evnt));
for j = 1:length(evnt)

    % print info
    fprintf('%d ',j);
    timerStart = tic;

    % epoch data in windows
%     d_wind = all_data_ds(:,round((evnt(j).CorrectStartTime-tmin)*new_fs):round((evnt(j).CorrectStartTime+tmax)*new_fs));
    d_sent = all_data_ds(:,round((evnt(j).CorrectStartTime-tmin)*new_fs):round((evnt(j).StopTime+tmax)*new_fs));

    % epoch data in windows
    sua_sent = spikes_sua(:,round((evnt(j).CorrectStartTime-tmin)*new_fs):round((evnt(j).StopTime+tmax)*new_fs));
    mua_sent = spikes_mua(:,round((evnt(j).CorrectStartTime-tmin)*new_fs):round((evnt(j).StopTime+tmax)*new_fs));

    % collect in struct
    out(j).resp = d_sent; %d_sent(sort_idx, :);
    out(j).tmin = -tmin; out(j).tmax = tmin;  % symmetric 
    out(j).dataf = new_fs;
    out(j).chs_original = chs_original;
    out(j).chs_physical = sorted_vals;
    
    fprintf(1,'Loading %s\n',evnt(j).wname);
    [t1,fw] = audioread(evnt(j).wname); % load audio file of sound
    
    % If the sound is stereo, make it mono only
    if (size(t1,2)>1)
        t1=t1(:,1);
    end
    % Resample to 16 kHz
%     t1 = resample(t1,16000,fw);
%     fw = 16000;
        
    % epoch anin
%     anin_wind = aninArray(:,round((evnt(j).CorrectStartTime-tmin)*nSamp_anin):round((evnt(j).CorrectStartTime+tmax)*nSamp_anin));
    anin_sent = aninArray(:,round((evnt(j).CorrectStartTime-tmin)*nSamp_anin):round((evnt(j).StopTime+tmax)*nSamp_anin));

    % aud features
    [env, peakRate, peakEnv] = find_peakRate(anin_sent(3, :), nSamp_anin);

    % Add zero padding on either side as appropriate
    t1 = [zeros(tmin*fw,1); t1(ceil(abs(tdiff(j))*fw):end) ;zeros(tmin*fw,1)];
    t1 = resample(t1,new_fs,fw);
    out(j).soundf = fw;
    out(j).sound = t1;
    wintime = 0.025;
    steptime = 1/fw;
    pspectrum = powspec(t1, fw, wintime, steptime);
    aspectrum = audspec(pspectrum, fw, 80, 'mel');
    aud = aspectrum.^( 0.077);
    out(j).aud  = aud;
    out(j).duration = length(t1)/fw;

    out(j).envelope = env;
    out(j).peakRate = peakRate;
    out(j).peakEnv = peakEnv;

    % add zscore and spikes
    %out_spikes(j).zscore_resp = zscore(double(out(j).resp)')';
    zscore_d = zscore(double(out(j).resp)')';

    % get spike data
    evnt_thresh = 3; % 3 sd's
    thresh_dat = abs(zscore_d);
    thresh_dat = thresh_dat > evnt_thresh;
    spk_data = zeros(size(thresh_dat));
    for i = 1:size(thresh_dat,1)
        tmp = findstr([0 thresh_dat(i,:)], [0 1]);
        tmp(find(diff(tmp) <= 0.001*out(j).dataf)) = [];
        spk_data(i,tmp) = 1;
    end
    %out_spikes(j).spikes = spk_data;
    spike_ev(j, :, :) = spk_data(:, 1:2*new_sfreq);

    % get the downsampled
    x_ds = resample(spk_data', 1, 10)';

    % find spike w threshold
    thresh = mean(max(x_ds) / 2);
    x_ds_t = x_ds > thresh;
    spike_ds(j, :, :) = x_ds_t(:, 1:2000);
    out(j).spikes = x_ds_t;

    % check correlation between number of spikes in orig and ds
    rs = corrcoef(sum(spike_ev(j, :, :), 3), sum(spike_ds(j, :, :), 3));
    corrcoefs(j) = rs(2);

    % do the downsampling for the sorted spikes too
    spk_data = zeros(size(sua_sent));
    for i = 1:size(sua_sent, 1)
        tmp = findstr([0 sua_sent(i,:)], [0 1]);
        tmp(find(diff(tmp) <= 0.001*out(j).dataf)) = [];
        spk_data(i,tmp) = 1;
    end

    % get the downsampled
    x_ds = resample(spk_data', 1, 10)';

    % find spike w threshold
    thresh = mean(max(x_ds) / 2);
    x_ds_t = x_ds > thresh;
    out(j).spikes_sua = x_ds_t;


    % do the downsampling for the sorted spikes too
    spk_data = zeros(size(mua_sent));
    for i = 1:size(mua_sent, 1)
        tmp = findstr([0 mua_sent(i,:)], [0 1]);
        tmp(find(diff(tmp) <= 0.001*out(j).dataf)) = [];
        spk_data(i,tmp) = 1;
    end

    % get the downsampled
    x_ds = resample(spk_data', 1, 10)';

    % find spike w threshold
    thresh = mean(max(x_ds) / 2);
    x_ds_t = x_ds > thresh;
    out(j).spikes_mua = x_ds_t;

    % matrix
%     evoked(j, :, :) = zscore(double(d_wind(sort_idx, :)));
%     evoked_audio(j, :) = anin_wind(3, :);

    % how long did that take?
    toc(timerStart);
end


%%

%save final results using v7 to load w python. have to use 7.3 for big
%files
fprintf('Saving final results in .mat file  \n')
make_fname = strcat('ks_preproc_out_', string(new_sfreq), '.mat');
fname = fullfile(rootZ, make_fname);
save(fname, 'out', '-v7.3');

% make_fname = strcat('ks_preproc_spikes_', string(new_sfreq), '.mat');
% fname = fullfile(rootZ, make_fname);
% save(fname, 'out_spikes', '-v7');
% 
% make_fname = strcat('lg-ks_sua_', string(new_sfreq), '.mat');
% fname = fullfile(rootZ, make_fname);
% save(fname, 'out_sua', '-v7');
% 
% make_fname = strcat('lg-ks_mua_', string(new_sfreq), '.mat');
% fname = fullfile(rootZ, make_fname);
% save(fname, 'out_mua', '-v7');

%% helper functions



% =========================================================
% Return sample rate as double.
%
function srate = SampRate(meta)
    if strcmp(meta.typeThis, 'imec')
        srate = str2double(meta.imSampRate);
    else
        srate = str2double(meta.niSampRate);
    end
end % SampRate

function [meta] = ReadMeta(binName, path)

    % Create the matching metafile name
    [dumPath,name,dumExt] = fileparts(binName);
    metaName = strcat(name, '.meta');

    % Parse ini file into cell entries C{1}{i} = C{2}{i}
    fid = fopen(fullfile(path, metaName), 'r');
% -------------------------------------------------------------
%    Need 'BufSize' adjustment for MATLAB earlier than 2014
%    C = textscan(fid, '%[^=] = %[^\r\n]', 'BufSize', 32768);
    C = textscan(fid, '%[^=] = %[^\r\n]');
% -------------------------------------------------------------
    fclose(fid);

    % New empty struct
    meta = struct();

    % Convert each cell entry into a struct entry
    for i = 1:length(C{1})
        tag = C{1}{i};
        if tag(1) == '~'
            % remake tag excluding first character
            tag = sprintf('%s', tag(2:end));
        end
        meta = setfield(meta, tag, C{2}{i});
    end
end % ReadMeta


% =========================================================
% Read nSamp timepoints from the binary file, starting
% at timepoint offset samp0. The returned array has
% dimensions [nChan,nSamp]. Note that nSamp returned
% is the lesser of: {nSamp, timepoints available}.
%
% IMPORTANT: samp0 and nSamp must be integers.
%
function dataArray = ReadBin(samp0, nSamp, meta, binName, path)

    nChan = str2double(meta.nSavedChans);

    nFileSamp = str2double(meta.fileSizeBytes) / (2 * nChan);
    samp0 = max(samp0, 0);
    nSamp = min(nSamp, nFileSamp - samp0);

    sizeA = [nChan, nSamp];

    fid = fopen(fullfile(path, binName), 'rb');
    fseek(fid, samp0 * 2 * nChan, 'bof');
    dataArray = fread(fid, sizeA, 'int16=>double');
    fclose(fid);
end % ReadBin
