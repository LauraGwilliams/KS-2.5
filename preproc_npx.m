%% define params

%% provide subject and block through indices
subj_n = 2;
block_n = 2;
decimation = 30;

%% init of items not requiring update re blocks/subjects

% first, load the config file
define_cfg;

% second, add to path
addpath(genpath(base_dir)) % add all softwares and datas
timitPath = '/home/lgwilliams/matlab/TIMIT';
addpath(genpath(timitPath));

% config file
pathToYourConfigFile = strcat(base_dir, '/scripts/configFiles'); % take from Github folder and put it somewhere else (together with the master_file)
pathToYourConfigFile = '/userdata/lgwilliams/neuropixels/software/KS-2.5/configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

%% init items that depend on block/subject

% basics
subject = cfg(subj_n).subject;
block = cfg(subj_n).blocks(block_n);
chanMapFile = cfg(subj_n).chanMapFile;
ks_dir = cfg(subj_n).ks_dir;
ks_csv = char(strcat(ks_dir, '/', cfg(subj_n).ks_csv));
ks_sorters = cfg(subj_n).ks_sorters;
crops = cell2mat(cfg(subj_n).crop_times(block_n));
crop_start = crops(1);
crop_stop = crops(2);

% is this is the result of 2+ recordings concatenated?
if strcmp(block, 'tcat')
    tname = 'tcat';
else
    tname = 't0';
end

% has spike sorting been done?
ks_flag = ~isempty(cfg(subj_n).ks_csv);

% paths
raw_dir = cfg(subj_n).raw_folder;
folder_name = strcat(subject, '_', block, '_g0');
imec_name = strcat(folder_name, '_imec0');
root = strcat(raw_dir, folder_name);
rootZ = char(strcat(root, '/', imec_name)); % the raw data binary file is in this folder

% params
ops.trange    = cell2mat(cfg(subj_n).crop_times(block_n)); % time range to sort
ops.NchanTOT  = cfg(subj_n).NchanTOT; % total number of channels in your recording

% deriv paths
ops.fproc   = fullfile(rootZ, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = fullfile(pathToYourConfigFile, chanMapFile);

anin_path = char(root);
aninBin_fname = char(strcat(folder_name, '_', tname, '.nidq.bin'));
aninMeta_fname = char(strcat(folder_name, '_', tname, '.nidq.meta'));

%% load raw lfp
lfp_fname = char(strcat(folder_name, '_', tname, '.imec0.lf.bin'));
meta_fname = char(strcat(folder_name, '_', tname, '.imec0.lf.meta'));
meta = ReadMeta(meta_fname, rootZ);
lfp_sr = floor(str2double(meta.imSampRate));
t0 = round(lfp_sr*ops.trange(1));
n_samples_in_seconds = ops.trange(2) - ops.trange(1);
t1 = round(lfp_sr*n_samples_in_seconds);
lfp_data = ReadBin(t0,t1,meta,lfp_fname,rootZ);

% apply gain correction
lfp_data = GainCorrectIM(lfp_data, 1:4, meta);

%% load params

% load the channel map
load(ops.chanMap);
connected_orig = connected; % will need this for the full probe stuff

%% load events
evnt_fname = char(strcat(subject, '_', block, '_evnt_CORRECTED.csv'));
evnt = table2struct(readtable(fullfile(rootZ, evnt_fname)));

% fix evnt paths
for i = 1:length(evnt)
    name = evnt(i).wname;
    tmp = strsplit(name,'@');
    evnt(i).wname = [timitPath '/@' tmp{2}];
end


%% channels
% load the true channel ordering
ch_struct = load('npx_chMap.mat');
value_cut = ch_struct.chMap(find(connected == 0));
ops.connected = connected;

%% this block rruns all the steps of the algorithm
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
spk_fs_after_decim = sfreq / decimation;

% get data
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
phn_fname = char(strcat(subject, '_', block', '_phon_events.csv'));
phn_evnt = table2struct(readtable(fullfile(rootZ, phn_fname)));

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

%% ok, load the channel map we used
load(ops.chanMap); % function to load channel map file
[depths384, depth_order384] = sort(ycoords);
[depths383, depth_order383] = sort(ycoords(connected));
            
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


%% derive params

% this is only applicable if ks spike sorting has been done
if ks_flag

% read the cluster allocations
cluster_alloc = table2struct(readtable(ks_csv));

% because of the batching, we might end up with more time samples than we
% asked for. we will see later how we want to deal with this. but for now,
% lets just ignore any sample which is longer than the crop max.
desired_max_tsamp = sfreq * n_samples_in_seconds;

% 1) init the spike train matrix, just for the sua for now
n_sua = length(cluster_alloc);
spikes_sua = zeros(n_sua, desired_max_tsamp);

% lets also make a new cluster matrix with just the sua, but across all
% users, and change the id to the new unique 97 ids
% no, that doesnt stupid work.... because they will be differences sizes
% and whatever.... maybe? 
spike_clusters_new_id = [];
spike_times_new_id = [];

% loop through each "user" and add the clusters defined for that person
for researcher = ks_sorters
    
    disp(researcher{1});
    
    % OK, i have to loop through researchers because that is the
    % correspondance between the cluster csv and the kilosort output of
    % cluster list. 
    
    % get the cluster ids that are relevant to this user..?
    r_id_list = [cluster_alloc.id];
    r_idx_list = [cluster_alloc.sua_id];
    user_idx = strcmp({cluster_alloc.user}, researcher{1});
    r_id_list = r_id_list(user_idx);
    cluster_indices = r_idx_list(user_idx); % possible indices
    
    % load the clusters of this person
    ks_output_dir = strcat(ks_dir, '/spike_sorted-', researcher);
    spike_times_fname = fullfile(ks_output_dir, 'spike_times.npy');
    spike_clusters_fname = fullfile(ks_output_dir, 'spike_clusters.npy');
    
    % load the spike sorted output from the .npy files
    % read in the files
    spike_times = readNPY(spike_times_fname{1});
    spike_clusters = readNPY(spike_clusters_fname{1});
    n_spike_events = length(spike_times);
    max_spike_time = max(spike_times);
    n_clusters = length(unique(spike_clusters));
    
    % 2) loop through each spike event and put into the correct bin
    for ev = 1:n_spike_events

        % get time and label
        s_time = spike_times(ev);
        s_label = spike_clusters(ev);

        % only if this is the cluster we picked
        if sum(find(s_label == r_id_list)) > 0

            % the batching issue - maybe lets fix this
            if s_time < desired_max_tsamp

            % get identity - can be done for multiple suas at a time
            idx_sua = find(r_id_list == s_label);
            cluster_idx = cluster_indices(idx_sua);
            
            % put into correct place
            if sum(idx_sua) > 0
                spikes_sua(cluster_idx, s_time) = 1;
                
                % add to the re-allocated cluster id and times
                spike_clusters_new_id = [spike_clusters_new_id, cluster_idx];
                spike_times_new_id = [spike_times_new_id, s_time];
                
            end

            disp(s_time);
            end
            
            
        else
            % TODO: add the mua here
           
            
        end
    end
    
    % get the waveforms
    
end

% organise the full clusters by time to be contiguous
[sort_vals, sort_idx_c] = sort(spike_times_new_id);
spike_times_new_id = spike_times_new_id(sort_idx_c);
spike_clusters_new_id = spike_clusters_new_id(sort_idx_c);
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
    s_dur = evnt(j).StopTime - sec_start;
    sec_stop = sec_start + s_dur;
    
    % get sentence duration
    sec_dur = sec_stop - sec_start;
    
    % descrepency between wav start and phoneme start
    this_tdiff = evnt(j).CorrectedVal;
    
    % pad start and stop
    pad_start = sec_start - tmin;
    pad_stop = sec_stop + tmax;

    % print info
    fprintf('%d ',j);
    timerStart = tic;

    % epoch data around sentence
    d_sent = all_data_ds(depth_order383, round((pad_start)*new_fs):round((pad_stop)*new_fs));
    
    % carefulllllll! 
    lfp_sent = lfp_data(depth_order384, round((pad_start)*lfp_sr):round((pad_stop)*lfp_sr));
    % remove the same channel that is missing in hp
    lfp_sent = lfp_sent(connected(depth_order384), :);
    
    % epoch data in windows
    if ks_flag
    sua_sent = spikes_sua(:,round((pad_start)*new_fs):round((pad_stop)*new_fs));
    end
    
    % collect in struct
    
    % downsample the high passed resp
    resp_ds = resample(double(d_sent)', spk_fs_after_decim, new_fs)';
    lfp_ds = resample(double(lfp_sent)', spk_fs_after_decim, lfp_sr)';
    out(j).lfp_ds = zscore(double(lfp_ds)')';
    out(j).resp_ds = zscore(double(resp_ds)')';
    
    %out(j).lfp_ds_not_z = lfp_ds;
    %out(j).lfp_matt = zscore(double(lfp_sent)')';
    
    %out(j).lfp_ds_gain = resample(double(lfp_sent_gain)', spk_fs_after_decim, lfp_sr)';
    
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
    t1 = [zeros(tmin*fw,1); t1(ceil(abs(this_tdiff)*fw):end) ;zeros(tmax*fw,1)];

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

    % NEW METHOD
    if ks_flag
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

    % time that the sentence started and stopped, in SAMPLES
    trlTimes = round((pad_start)*new_fs):round((pad_stop)*new_fs);
    
    % get INDEX when the spikes occurred during the sentence
    trlSpikes = find((spike_times_new_id >= trlTimes(1) & (spike_times_new_id <= trlTimes(end))));
    
    % subset spike events (cluster id list) based on indices
    % ie which spikes fired during the sentence?
    trlSpikeClusts = spike_clusters_new_id(trlSpikes);

    clear spk_wave
    
    % for each cluster index
    for cluster_index = 1:length(cluster_alloc)
        % find indices of that id
        %idx = find(trlSpikeClusts == sua_clust_ids(cluster_index));
        idx = find(trlSpikeClusts == cluster_index);
        % for that spike event
        for event_n = 1:length(idx)
            % crop around the spike time and store it
            
            % 1) get data for central channel
            %ch_n = sua_info(cluster_index).ch+1;
            ch_n = cluster_alloc(cluster_index).ch+1;
            
            % 2) get INDEX of central spike for that spike event
            % this is in indices not time, relative to spike_times
            c_idx = trlSpikes(idx(event_n));
            c_t = spike_times_new_id(c_idx);
            
            % 3) get 1 ms before and after
            c_t_min = c_t - (0.001*new_fs);
            c_t_max = c_t + (0.001*new_fs);
            
            % 4) crop and store
            spk_wave{cluster_index}(event_n, :) = double(all_data_ds(ch_n, c_t_min:c_t_max));
        end
    end
    out(j).spike_wave = spk_wave;
    end
    
    % add the phonetic and phonemic information
    sentence_label = out(j).name;
    phn_tc = zeros(length(phn_cats), size(out(j).spikes, 2));
    feature_tc = zeros(length(phonetic_features)+1, size(out(j).spikes, 2));
    for phn_i = 1:length(phn_evnt)
        
        % only add if it is the right sentence 
        if strcmp(sentence_label, phn_evnt(phn_i).sent_name)
            
            % if its the first phoneme, we want to correct this value
            if phn_i == 1
                correction_val = floor(phn_evnt(phn_i).start / 16);
            end

            % phn onset
            onset = floor(phn_evnt(phn_i).start / 16) - correction_val;
            offset = floor(phn_evnt(phn_i).stop / 16) - correction_val;
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
if ks_flag
out(1).sua_info = cluster_alloc; % from the csv
end

% have to use 7.3 for big files
fprintf('Saving final results in .mat file  \n')
make_fname = strcat(subject, '_', block', '_preproc.mat');
fname = char(fullfile(save_dir, make_fname));
save(fname, 'out', '-v7.3');

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





% =========================================================
% Return an array [lines X timepoints] of uint8 values for
% a specified set of digital lines.
%
% - dwReq is the one-based index into the saved file of the
%    16-bit word that contains the digital lines of interest.
% - dLineList is a zero-based list of one or more lines/bits
%    to scan from word dwReq.
%
function digArray = ExtractDigital(dataArray, meta, dwReq, dLineList)
    % Get channel index of requested digital word dwReq
    if strcmp(meta.typeThis, 'imec')
        [AP, LF, SY] = ChannelCountsIM(meta);
        if SY == 0
            fprintf('No imec sync channel saved\n');
            digArray = [];
            return;
        else
            digCh = AP + LF + dwReq;
        end
    else
        [MN,MA,XA,DW] = ChannelCountsNI(meta);
        if dwReq > DW
            fprintf('Maximum digital word in file = %d\n', DW);
            digArray = [];
            return;
        else
            digCh = MN + MA + XA + dwReq;
        end
    end
    [~,nSamp] = size(dataArray);
    digArray = zeros(numel(dLineList), nSamp, 'uint8');
    for i = 1:numel(dLineList)
        digArray(i,:) = bitget(dataArray(digCh,:), dLineList(i)+1, 'int16');
    end
end % ExtractDigital

% =========================================================
% Return a multiplicative factor for converting 16-bit
% file data to voltage. This does not take gain into
% account. The full conversion with gain is:
%
%   dataVolts = dataInt * fI2V / gain.
%
% Note that each channel may have its own gain.
%
function fI2V = Int2Volts(meta)
    if strcmp(meta.typeThis, 'imec')
        fI2V = str2double(meta.imAiRangeMax) / 512;
    else
        fI2V = str2double(meta.niAiRangeMax) / 32768;
    end
end % Int2Volts


% =========================================================
% Return array of original channel IDs. As an example,
% suppose we want the imec gain for the ith channel stored
% in the binary data. A gain array can be obtained using
% ChanGainsIM() but we need an original channel index to
% do the look-up. Because you can selectively save channels
% the ith channel in the file isn't necessarily the ith
% acquired channel, so use this function to convert from
% ith stored to original index.
%
% Note: In SpikeGLX channels are 0-based, but MATLAB uses
% 1-based indexing, so we add 1 to the original IDs here.
%
function chans = OriginalChans(meta)
    if strcmp(meta.snsSaveChanSubset, 'all')
        chans = (1:str2double(meta.nSavedChans));
    else
        chans = str2num(meta.snsSaveChanSubset);
        chans = chans + 1;
    end
end % OriginalChans


% =========================================================
% Return counts of each imec channel type that compose
% the timepoints stored in binary file.
%
function [AP,LF,SY] = ChannelCountsIM(meta)
    M = str2num(meta.snsApLfSy);
    AP = M(1);
    LF = M(2);
    SY = M(3);
end % ChannelCountsIM

% =========================================================
% Return counts of each nidq channel type that compose
% the timepoints stored in binary file.
%
function [MN,MA,XA,DW] = ChannelCountsNI(meta)
    M = str2num(meta.snsMnMaXaDw);
    MN = M(1);
    MA = M(2);
    XA = M(3);
    DW = M(4);
end % ChannelCountsNI


% =========================================================
% Return gain for ith channel stored in the nidq file.
%
% ichan is a saved channel index, rather than an original
% (acquired) index.
%
function gain = ChanGainNI(ichan, savedMN, savedMA, meta)
    if ichan <= savedMN
        gain = str2double(meta.niMNGain);
    elseif ichan <= savedMN + savedMA
        gain = str2double(meta.niMAGain);
    else
        gain = 1;
    end
end % ChanGainNI


% =========================================================
% Return gain arrays for imec channels.
%
% Index into these with original (acquired) channel IDs.
%
function [APgain,LFgain] = ChanGainsIM(meta)

    if isfield(meta,'imDatPrb_dock')
        [AP,LF,~] = ChannelCountsIM(meta);
        % NP 2.0; APgain = 80 for all channels
        APgain = zeros(AP,1,'double');
        APgain = APgain + 80;
        % No LF channels, set gain = 0
        LFgain = zeros(LF,1,'double');
    else
        % 3A or 3B data?
        % 3A metadata has field "typeEnabled" which was replaced
        % with "typeImEnabled" and "typeNiEnabled" in 3B.
        % The 3B imro table has an additional field for the
        % high pass filter enabled/disabled
        if isfield(meta,'typeEnabled')
            % 3A data
            C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d', ...
                'EndOfLine', ')', 'HeaderLines', 1 );
        else
            % 3B data
            C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d %*s', ...
                'EndOfLine', ')', 'HeaderLines', 1 );
        end
        APgain = double(cell2mat(C(1)));
        LFgain = double(cell2mat(C(2)));
    end
end % ChanGainsIM


% =========================================================
% Having acquired a block of raw nidq data using ReadBin(),
% convert values to gain-corrected voltages. The conversion
% is only applied to the saved-channel indices in chanList.
% Remember saved-channel indices are in range [1:nSavedChans].
% The dimensions of the dataArray remain unchanged. ChanList
% examples:
%
%   [1:MN]      % all MN chans (MN from ChannelCountsNI)
%   [2,6,20]    % just these three channels
%
function dataArray = GainCorrectNI(dataArray, chanList, meta)

    [MN,MA] = ChannelCountsNI(meta);
    fI2V = Int2Volts(meta);

    for i = 1:length(chanList)
        j = chanList(i);    % index into timepoint
        conv = fI2V / ChanGainNI(j, MN, MA, meta);
        dataArray(j,:) = dataArray(j,:) * conv;
    end
end


% =========================================================
% Having acquired a block of raw imec data using ReadBin(),
% convert values to gain-corrected voltages. The conversion
% is only applied to the saved-channel indices in chanList.
% Remember saved-channel indices are in range [1:nSavedChans].
% The dimensions of the dataArray remain unchanged. ChanList
% examples:
%
%   [1:AP]      % all AP chans (AP from ChannelCountsIM)
%   [2,6,20]    % just these three channels
%
function dataArray = GainCorrectIM(dataArray, chanList, meta)

    % Look up gain with acquired channel ID
    chans = OriginalChans(meta);
    [APgain,LFgain] = ChanGainsIM(meta);
    nAP = length(APgain);
    nNu = nAP * 2;

    % Common conversion factor
    fI2V = Int2Volts(meta);

    for i = 1:length(chanList)
        j = chanList(i);    % index into timepoint
        k = chans(j);       % acquisition index
        if k <= nAP
            conv = fI2V / APgain(k);
        elseif k <= nNu
            conv = fI2V / LFgain(k - nAP);
        else
            continue;
        end
        dataArray(j,:) = dataArray(j,:) * conv;
    end
end