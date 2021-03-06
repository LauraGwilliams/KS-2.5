% global script for generating TIMIT out structures.

%% provide subject and block through indices
subj_n = 1;
block_n = 1;

%% init of items not requiring update re blocks/subjects

% first, load the config file
define_cfg;

% second, add to path
addpath(genpath(base_dir)) % add all softwares and datas

% config file
pathToYourConfigFile = strcat(base_dir, '/scripts/configFiles'); % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

%% init items that depend on block/subject

% basics
subject = cfg(subj_n).subject;
block = cfg(subj_n).blocks(block_n);
chanMapFile = cfg(subj_n).chanMapFile;
crops = cell2mat(cfg(subj_n).crop_times(block_n));
crop_start = crops(1);
crop_stop = crops(2);

% paths
raw_dir = cfg(subj_n).raw_folder;
folder_name = strcat(subject, '_', block, '_g0');
imec_name = strcat(folder_name, '_imec0');
root = strcat(raw_dir, folder_name);
rootZ = char(strcat(root, '/', imec_name)); % the raw data binary file is in this folder

% paths to ks
subj_dir = cfg(subj_n).subj_folder;
ks_output_dir = char(strcat(subj_dir, '/spike_sorted-ks'));

% params
ops.trange    = cell2mat(cfg(subj_n).crop_times(block_n)); % time range to sort
ops.NchanTOT  = cfg(subj_n).NchanTOT; % total number of channels in your recording

% deriv paths
ops.fproc   = fullfile(rootZ, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = fullfile(pathToYourConfigFile, chanMapFile);

anin_path = char(root);
aninBin_fname = char(strcat(folder_name, '_t0.nidq.bin'));
aninMeta_fname = char(strcat(folder_name, '_t0.nidq.meta'));


%% load params

% load the channel map
load(ops.chanMap);

% load CORRECTED! event structure
%events = load(fullfile(rootZ, 'NP04_evnt.mat'));
%evnt = events.evnt;
evnt_file = char(strcat(rootZ, '/', subject, '_', block, '_evnt_CORRECTED.csv'));
evnt = table2struct(readtable(evnt_file));

% load the true channel ordering
root_np4 = '/userdata/lgwilliams/neuropixels/data/NP04/raw/NP04_B2_g0/NP04_B2_g0_imec0';
ch_struct = load(strcat(root_np4, '/NP04_chMap.mat'));
value_cut = ch_struct.chMap(find(connected == 0));

% which channels we use are defined in the physical space of the probe, not
% the chronological order of the data
bad_idx = find(ch_struct.chMap < 10 | ch_struct.chMap > 320);
connected(bad_idx) = 0;
connected(find(ch_struct.chMap == 186)) = 0;
connected(find(ch_struct.chMap == 187)) = 0;
ops.connected = connected;

% define path to spike data
spike_times_fname = fullfile(ks_output_dir, 'spike_times.npy');
spike_clusters_fname = fullfile(ks_output_dir, 'spike_clusters.npy');

%% load params

% load the channel map
load(ops.chanMap);

% load CORRECTED! event structure
evnt_file = char(strcat(rootZ, '/', subject, '_', block, '_evnt_CORRECTED.csv'));
evnt = table2struct(readtable(evnt_file));

% fix evnt paths
for i = 1:length(evnt)
    name = evnt(i).wname;
    tmp = strsplit(name,'@');
    evnt(i).wname = [timitPath '/@' tmp{2}];
end

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
ops.fbinary = fullfile(rootZ, fs(find(contains({fs.name},'ap'))).name);

% preprocess data to create temp_wh.dat
sfreq = 30000;
new_sfreq = sfreq;
decimation = 1; % do not apply.
spk_fs_after_decim = sfreq / decimation;

% fprintf('Loading preproc binary....\n');
% meta = ReadMeta('NP04_B2_gnew.meta',rootZ);
% t0 = round(str2double(meta.imSampRate)*240);
% t1 = round(str2double(meta.imSampRate)*600);
% dataArray = ReadBin(t0,t1,meta,'NP04_B2_gnew.bin',rootZ);

% get data
%load('all_data_ds.mat');
all_data_ds = preprocessData_returnMatrix(ops);

%% load anin

% load it
aninmeta = ReadMeta(aninMeta_fname, anin_path);

% Get first one second of data
nSamp_anin = floor(1.0 * SampRate(aninmeta));

% timing
t0 = round(str2double(aninmeta.niSampRate)*crop_start);
n_samples_in_seconds = crop_stop-crop_start;
t1 = round(str2double(aninmeta.niSampRate)*n_samples_in_seconds);
aninArray = ReadBin(t0,t1,aninmeta,aninBin_fname,anin_path);


%% load phonetic feature times

% get the phonetic info
phn_evnt = table2struct(readtable(fullfile(rootZ, 'NP04_phon_events.csv')));

% get the phoneme categories
phn_cats = [];
for ii = 1:length(phn_evnt)
   phn_cats = [phn_cats; string(phn_evnt(ii).phoneme)];
end
phn_cats = unique(phn_cats);

% define the features of interest
phonetic_features = ["word_onset"; "phonation_v"; "manner_o"; "manner_a"; ...
                     'manner_v'; 'manner_f'; 'manner_n'; "place_l"; ...
                     "place_v"; "place_h"; "place_c"; "place_g"; ...
                     "place_lo"; "place_m"; "place_d"; "frontback_f"; ...
                     "frontback_b";"frontback_n"; "roundness_u"; ...
                     "roundness_r"; "centrality_n"; "centrality_f"; ...
                     "centrality_c"];

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
%spike_times = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/spike_times.npy');
%spike_clusters = readNPY('/userdata/lgwilliams/neuropixels/data/NP04/spike_sorted-lg/spike_clusters.npy');

% read in the files
spike_times = readNPY(spike_times_fname);
spike_clusters = readNPY(spike_clusters_fname);

% read in the cluster labels and note the sua and mua cluster ids
mua_label = [];
sua_label = [];
cluster_info = table2struct(readtable(fullfile(ks_output_dir, 'cluster_info.csv')));

% init
sua_counter = 1; mua_counter = 1;
clear sua_info
clear mua_info

% loop
for row_i = 1:length(cluster_info)
    
    % is this a sua cluster?
    if length(cluster_info(row_i).final) > 0
        sua_label = [sua_label, cluster_info(row_i).id];
        
        % pull out all the info from the cluster csv
        % OK this code is so ugly plz fix
        sua_info(sua_counter).id = cluster_info(row_i).id;        
        sua_info(sua_counter).depth = cluster_info(row_i).depth;
        sua_info(sua_counter).fr = cluster_info(row_i).fr;
        sua_info(sua_counter).ch = cluster_info(row_i).ch;
        sua_info(sua_counter).amp = cluster_info(row_i).Amplitude;
        sua_info(sua_counter).n_spikes = cluster_info(row_i).n_spikes;
        sua_info(sua_counter).group = 'sua';
        sua_counter = sua_counter + 1;
   
    % otherwise, is this a mua cluster?
    % ok this is a dumb way but i'm not sure how else -- 3 chars = "MUA"
    elseif length(cluster_info(row_i).group) == 3
        mua_label = [mua_label, cluster_info(row_i).id];

        % pull out all the info from the cluster csv
        % OK this code is so ugly plz fix
        mua_info(mua_counter).id = cluster_info(row_i).id;        
        mua_info(mua_counter).depth = cluster_info(row_i).depth;
        mua_info(mua_counter).fr = cluster_info(row_i).fr;
        mua_info(mua_counter).ch = cluster_info(row_i).ch;
        mua_info(mua_counter).amp = cluster_info(row_i).Amplitude;
        mua_info(mua_counter).n_spikes = cluster_info(row_i).n_spikes;
        mua_info(mua_counter).group = 'mua';
        mua_counter = mua_counter + 1;        
        
    end
end
sua_clust_ids = [sua_info.id];


%% derive params
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
    
    % get tmin and tmax for this sentence
    sec_start = evnt(j).CorrectStartTime;
    sec_stop = evnt(j).StopTime;
    
    % get sentence duration
    sec_dur = sec_stop - sec_start;
    
    % descrepency between wav start and phoneme start
    this_tdiff = evnt(j).CorrectStartTime - evnt(j).StartTime;
    
    % pad start and stop
    pad_start = sec_start - tmin;
    pad_stop = sec_stop + tmax;

    % print info
    fprintf('%d ',j);
    timerStart = tic;

    % epoch data around sentence
    d_sent = all_data_ds(:,round((pad_start)*new_fs):round((pad_stop)*new_fs));

    % epoch data in windows
    sua_sent = spikes_sua(:,round((pad_start)*new_fs):round((pad_stop)*new_fs));
    mua_sent = spikes_mua(:,round((pad_start)*new_fs):round((pad_stop)*new_fs));

    % collect in struct
    %out(j).resp = d_sent; %d_sent(sort_idx, :);
    out(j).tmin = -tmin; out(j).tmax = tmax;  % not necc symmetric 
    %out(j).dataf = new_fs;
    out(j).spike_fs = spk_fs_after_decim;
    out(j).chs_original = chs_original;
    out(j).chs_physical = sorted_vals;
    out(j).duration = sec_dur;
    
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
    anin_sent = aninArray(:,round((pad_start)*nSamp_anin):round((pad_stop)*nSamp_anin));

    % aud features
    [env, peakRate, peakEnv] = find_peakRate(anin_sent(3, :), nSamp_anin);
    
    % what is the sample rate of these aud features?
    ratio = size(env, 1) / size(anin_sent, 2);
    peakRate_fs = ceil(nSamp_anin * ratio);
    
    % resample env and peakRate features to match the rest
    % resampling the binary is not super great - is taking the abs the
    % best?
    env_rs = resample(env, spk_fs_after_decim, peakRate_fs);
    peakRate_rs = abs(resample(peakRate, spk_fs_after_decim, peakRate_fs));
    peakEnv_rs = abs(resample(peakEnv, spk_fs_after_decim, peakRate_fs));

    % Add zero padding to waveform either side as appropriate
    switch user
        case 'lg'
            t1 = [zeros(tmin*fw,1); t1(ceil(abs(this_tdiff)*fw):end) ;zeros(tmax*fw,1)];
        case 'mkl'
            t1 = [zeros(tmin*fw,1); t1(ceil(abs(tdiff(j))*fw):end) ;zeros(tmax*fw,1)];
    end
    %t1 = resample(t1,new_fs,fw); % incorrect

    % spectrogram - run it on the full sample rate and downsample after
    wintime = 0.025;
    steptime = 1/fw;
    pspectrum = powspec(t1, fw, wintime, steptime);
    aspectrum = audspec(pspectrum, fw, 80, 'mel');
    aud = aspectrum.^( 0.077);
    
    % compute practical fw of aud spec
    fw_aud = ceil((size(pspectrum, 2) / size(t1, 1)) * fw);
    
    % resample it
    aud_rs = resample(aud', spk_fs_after_decim, fw_aud)';
    sound_rs = resample(t1', spk_fs_after_decim, fw)';

    % add to the struct
    out(j).soundf = spk_fs_after_decim;
    out(j).sound = sound_rs;
    out(j).aud  = aud_rs;
    %out(j).duration = length(t1)/fw; % incorrect
    
    % add these aud features - they have been resampled to match spike rate
    out(j).envelope = env_rs;
    out(j).peakRate = peakRate_rs;
    out(j).peakEnv = peakEnv_rs;

    % add zscore and spikes
    %out_spikes(j).zscore_resp = zscore(double(out(j).resp)')';
    zscore_d = zscore(double(d_sent)')';

    % get spike data - MANUAL MUA
    evnt_thresh = 3; % 3 sd's
    % OLD METHOD
    thresh_dat = abs(zscore_d);
    thresh_dat = thresh_dat > evnt_thresh;
    spk_data = zeros(size(thresh_dat));
    for i = 1:size(thresh_dat,1)
        tmp = findstr([0 thresh_dat(i,:)], [0 1]);
        tmp(find(diff(tmp) <= 0.001*new_fs)) = [];
        spk_data(i,tmp) = 1;
    end
    bin_size = 0.001*new_fs;
    window_idx = 1:bin_size:size(spk_data,2);
    window_idx = window_idx(1:end-1);
    counter = 1;
    clear sua_sent_bin
    for w = window_idx
        w_data = spk_data(:,w:w+bin_size-1);
        spk_data_bin(:,counter) = sum(w_data,2);
        counter = counter + 1;
    end
    out(j).spikes = spk_data_bin;
    %out_spikes(j).spikes = spk_data;
    %spike_ev(j, :, :) = spk_data(:, 1:2*new_sfreq);
    % NEW METHOD
%     thresh_dat = abs(zscore_d);
%     spk_data = zeros(size(thresh_dat));
%     for i = 1:size(thresh_dat,1)
%         [~,locs] = findpeaks(thresh_dat(i,:),'MinPeakHeight',evnt_thresh,'MinPeakDistance',0.001*new_fs);
%         spk_data(i,locs) = 1;
%     end
    % get the downsampled
%     x_ds = resample(spk_data', 1, decimation)';
% 
%     % find spike w threshold
%     thresh = mean(max(x_ds) / 2);
%     x_ds_t = x_ds > thresh;
%     %spike_ds(j, :, :) = x_ds_t(:, 1:2000);
%     out(j).spikes = x_ds_t;

    % check correlation between number of spikes in orig and ds
    %rs = corrcoef(sum(spike_ev(j, :, :), 3), sum(spike_ds(j, :, :), 3));
    %corrcoefs(j) = rs(2);
    
    % get spike data - KS SUA
    % OLD METHOD
    % do the downsampling for the sorted spikes too
%     spk_data = zeros(size(sua_sent));
%     for i = 1:size(sua_sent, 1)
%         tmp = findstr([0 sua_sent(i,:)], [0 1]);
%         tmp(find(diff(tmp) <= 0.001*new_fs)) = [];
%         spk_data(i,tmp) = 1;
%     end
% 
%     % get the downsampled
%     x_ds = resample(spk_data', 1, decimation)';
% 
%     % find spike w threshold
%     thresh = mean(max(x_ds) / 2);
%     x_ds_t = x_ds > thresh;
%     out(j).spikes_sua = x_ds_t;
    % NEW METHOD
    bin_size = 0.001*new_fs;
    window_idx = 1:bin_size:size(sua_sent,2);
    window_idx = window_idx(1:end-1);
    counter = 1;
    clear sua_sent_bin
    for w = window_idx
        w_data = sua_sent(:,w:w+bin_size-1);
        sua_sent_bin(:,counter) = sum(w_data,2);
        counter = counter + 1;
    end
    out(j).spikes_sua = sua_sent_bin;

    % get spike data - KS MUA
    % OLD METHOD
    % do the downsampling for the sorted spikes too
%     spk_data = zeros(size(mua_sent));
%     for i = 1:size(mua_sent, 1)
%         tmp = findstr([0 mua_sent(i,:)], [0 1]);
%         tmp(find(diff(tmp) <= 0.001*new_fs)) = [];
%         spk_data(i,tmp) = 1;
%     end
% 
%     % get the downsampled
%     x_ds = resample(spk_data', 1, decimation)';
% 
%     % find spike w threshold
%     thresh = mean(max(x_ds) / 2);
%     x_ds_t = x_ds > thresh;
%     out(j).spikes_mua = x_ds_t;

    % NEW METHOD
    bin_size = 0.001*new_fs;
    window_idx = 1:bin_size:size(mua_sent,2);
    window_idx = window_idx(1:end-1);
    counter = 1;
    clear mua_sent_bin
    for w = window_idx
        w_data = mua_sent(:,w:w+bin_size-1);
        mua_sent_bin(:,counter) = sum(w_data,2);
        counter = counter + 1;
    end
    out(j).spikes_mua = mua_sent_bin;

    % SPIKE WAVEFORMS
    
    % time that the sentence started and stopped, in SAMPLES
    trlTimes = round((pad_start)*new_fs):round((pad_stop)*new_fs);
    
    % get INDEX where the spikes occurred during the sentence
    trlSpikes = find((spike_times >= trlTimes(1) & (spike_times <= trlTimes(end))));
    
    % subset spike events (cluster id list) based on indices
    trlSpikeClusts = spike_clusters(trlSpikes);

    clear spk_wave
    
    % for each cluster index
    for cluster_index = 1:length(sua_clust_ids)
        % find indices of that id
        idx = find(trlSpikeClusts == sua_clust_ids(cluster_index));
        % for that spike event
        for event_n = 1:length(idx)
            % crop around the spike time and store it
            
            % 1) get data for central channel
            ch_n = sua_info(cluster_index).ch+1;
            
            % 2) get INDEX of central spike for that spike event
            % this is in indices not time, relative to spike_times
            c_idx = trlSpikes(idx(event_n));
            c_t = spike_times(c_idx);
            
            % 3) get 1 ms before and after
            c_t_min = c_t - (0.001*new_fs);
            c_t_max = c_t + (0.001*new_fs);
            
            % 4) crop and store
            spk_wave{cluster_index}(event_n, :) = double(all_data_ds(ch_n, c_t_min:c_t_max));
        end
    end
    out(j).spike_wave = spk_wave;
    
    % add the phonetic and phonemic information
    sentence_label = out(j).name;
    phn_tc = zeros(length(phn_cats), size(out(j).spikes_sua, 2));
    feature_tc = zeros(length(phonetic_features)+1, size(out(j).spikes_sua, 2));
    for phn_i = 1:length(phn_evnt)
        
        % only add if it is the right sentence 
        if strcmp(sentence_label, phn_evnt(phn_i).sent_name)

            % phn onset
            onset = floor(phn_evnt(phn_i).start / 16);
            offset = floor(phn_evnt(phn_i).stop / 16);
            dur = offset - onset;
            onset = onset + (tmin*spk_fs_after_decim); % add the 500 ms shift

            % add the phoneme category
            p_idx = find(phn_cats == phn_evnt(phn_i).phoneme);
            phn_tc(p_idx, onset:onset+dur) = 1;

            % get the fields
            fields = fieldnames(phn_evnt(phn_i));
            for fi = 1:length(fields)
                f = fields(fi);
                f_idx = find(f == phonetic_features);
                if length(f_idx) > 0
                    val = eval(string(strcat('phn_evnt(phn_i).', f)));
                    feature_tc(f_idx, onset:onset+dur) = val;
                end
            end

            % add sentence onset
            if phn_i == 1
                feature_tc(end, onset:onset+dur) = 1;
            end
        end
        
    end
    
    % add to out struct
    out(j).phn_tc = phn_tc;
    out(j).feature_tc = feature_tc;
    out(j).phn_cats = phn_cats;
    out(j).features = phonetic_features;
    
    % matrix
%     evoked(j, :, :) = zscore(double(d_wind(sort_idx, :)));
%     evoked_audio(j, :) = anin_wind(3, :);

    % how long did that take?
    toc(timerStart);
end

% add sua and mua info to the out 
out(1).sua_info = sua_info;
out(1).mua_info = mua_info;

%%

%save final results using v7 to load w python. have to use 7.3 for big
%files
fprintf('Saving final results in .mat file  \n')
make_fname = strcat('ks_preproc_out_', string(spk_fs_after_decim), '.mat');
fname = fullfile(save_dir, make_fname);
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