% save matlab out struct as a continuous matrix for easy loading into
% python

%% define path to out structure
%out_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/ks_preproc_out_1000.mat';
%out_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/117_preproc_out_1000.mat';
out_fname = '/userdata/lgwilliams/neuropixels/data/NP12/raw/NP12_B5_g0/NP12_B5_g0_imec0/ks_preproc_out_30000.mat';

rootZ = '/userdata/lgwilliams/neuropixels/data/NP12/raw/NP12_B5_g0/NP12_B5_g0_imec0';
%ecog_fname = '/userdata/lgwilliams/neuropixels/data/NP12/raw/ecog/EC237_TIMIT_HilbAA_70to150_8band_mel_out_resp.mat';

% load ecog
%load(ecog_fname);
%ecog = out;

% load the spikes
load(out_fname);

% get the phonetic info
phn_evnt = table2struct(readtable(fullfile(rootZ, 'NP12_B5_phon_events.csv')));

% load formant info
%formant_info = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/out_NP04_formants.mat';
%load(formant_info);


%% phn set up 

% get the phoneme categories
phn_cats = [];
for ii = 1:length(phn_evnt)
   phn_cats = [phn_cats; string(phn_evnt(ii).phoneme)];
end
phn_cats = unique(phn_cats);

% define the features of interest
phonetic_features = ["phonation_v"; "manner_o"; "manner_a"; ...
                     'manner_v'; 'manner_f'; 'manner_n'; "place_l"; ...
                     "place_v"; "place_h"; "place_c"; "place_g"; ...
                     "place_lo"; "place_m"; "place_d"; "frontback_f"; ...
                     "frontback_b";"frontback_n"; "roundness_u"; ...
                     "roundness_r"; "centrality_n"; "centrality_f"; ...
                     "centrality_c"; "surp"; "entropy"];

%% generate continous matrices

% params
n_sents = length(out);
n_mel = size(out(1).aud, 1);
concat_length = 643100;  % overshoot a bit 

% init files
ecog_tc = zeros(256, concat_length);
lfp_ch = zeros(384, concat_length);
hp_ch = zeros(383, concat_length);
mel_spec = zeros(n_mel, concat_length);
phn_tc = zeros(length(phn_cats), concat_length);
feature_tc = zeros(length(phonetic_features)+2, concat_length); % plus sentence and word onsets
word_probs = zeros(3, concat_length);

%% loop through and make res

smin = 1;
for si = 1:n_sents
    
    % we will need this later
    sentence_label = out(si).name;

    % get sentence data
    this_lfp = out(si).lfp_ds;
    this_hp = out(si).resp_ds;
    
    % get mel data - downsample: this is an error in making the out
    this_mel = out(si).aud;
    
    % determine the size of this sentence
    n_samps = size(this_lfp, 2) - 1;
    smax = smin + n_samps - 1;
    
    % add data to the preallocation
    lfp_ch(:, smin:smax) = this_lfp(:, 1:n_samps);
    hp_ch(:, smin:smax) = this_hp(:, 1:n_samps);
    mel_spec(:, smin:smax) = this_mel(:, 1:smax-smin+1);
    
    % add the phoneme stuff
    for phn_i = 1:length(phn_evnt)
        
        % the npx and mel have already been aligned to the first phoneme
        % not the beginning of the file, so we need to account for that
        if phn_evnt(phn_i).sentence_onset == 1
            first_phn_onset = floor(phn_evnt(phn_i).start / 16);
        end
        
        % only add if it is the right sentence 
        if strcmp(sentence_label, phn_evnt(phn_i).sent_name)

            % phn onset
            onset = floor(phn_evnt(phn_i).start / 16) - first_phn_onset;
            offset = floor(phn_evnt(phn_i).stop / 16) - first_phn_onset;
            dur = offset - onset;
            onset = onset + 500 + smin; % add the 500 ms shift and time since last one

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
            
            % add word onset
            if phn_evnt(phn_i).word_onset == 1
                feature_tc(end-1, onset:onset+dur) = 1;
                
                % add surprisal values
                word_probs(1, onset:onset+dur) = phn_evnt(phn_i).unigram;
                word_probs(2, onset:onset+dur) = phn_evnt(phn_i).trigram;
                word_probs(3, onset:onset+dur) = phn_evnt(phn_i).gpt2;
            end

            % add sentence onset
            if phn_evnt(phn_i).sentence_onset == 1
                feature_tc(end, onset:onset+dur) = 1;
            end
        end
        
    % collect the formant info
%     if strcmp(sentdet(si).name, out(si).name)
%         f = sentdet(si).formants;
%         formant_tc(:, smin:smax) = f(:, 1:n_samps);
%         
%         a pitch normalised version
%         for form = 1:4
%             formant_normed_tc(form, smin:smax) = formant_tc(form, smin:smax) / nanmean(formant_tc(form, smin:smax));
%         end
%     end
        
    end
        
    % update the smin
    smin = smax;
    
    % how we doin?
    disp(si);
        
end

% normalise the formants
% formant_tc = formant_tc / max(max(formant_tc));

% save it
save_mel_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_ks_mel-1000.mat';
save_hp_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_hp117-1000.mat';
save_lfp_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_lfp117-1000.mat';

save_fea_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_ks_fea-1000.mat';
save_phn_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_ks_phn-1000.mat';
% save_formant_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_ks_formant-1000.mat';
% save_formant_normed_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/ks_formant_normed-1000.mat';
%save_ecog_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/ecog-1000.mat';
save_probs_fname = '/userdata/lgwilliams/neuropixels/data/NP12/out_structs/NP12_B5_probs-1000.mat';

save(save_mel_fname, 'mel_spec', '-v7.3');
save(save_lfp_fname, 'lfp_ch', '-v7.3');
save(save_hp_fname, 'hp_ch', '-v7.3');

save(save_phn_fname, 'phn_tc', '-v7.3');
save(save_fea_fname, 'feature_tc', '-v7.3');
% save(save_formant_fname, 'formant_tc', '-v7');
% save(save_formant_normed_fname, 'formant_normed_tc', '-v7');
save(save_probs_fname, 'word_probs', '-v7.3');
