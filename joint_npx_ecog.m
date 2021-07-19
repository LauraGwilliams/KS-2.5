% save matlab out struct as a continuous matrix for easy loading into
% python

%% define path to out structure
%out_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_preproc_out_1000.mat';
out_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/sel_sua_preproc_out_1000.mat';

rootZ = '/userdata/lgwilliams/neuropixels/data/NP04/raw/NP04_B2_g0/NP04_B2_g0_imec0';
ecog_fname = '/userdata/lgwilliams/neuropixels/data/NP04/raw/ecog/EC237_TIMIT_HilbAA_70to150_8band_mel_out_resp.mat';

% load ecog
load(ecog_fname);
ecog = out;

% load the spikes
load(out_fname);

% get the phonetic info
phn_evnt = table2struct(readtable(fullfile(rootZ, 'NP04_phon_events.csv')));

% load formant info
formant_info = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/out_NP04_formants.mat';
load(formant_info);


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

%% determine which are the sentences shared by npx and ecog
ecog_sents = {ecog.name};
npx_sents = {out.name};

% mm this is gonna be dumb but wtv
shared_sents = [];
for sn = npx_sents
    if any(strcmp(ecog_sents, sn{1})) % check it matches
        if ~any(strcmp(shared_sents, sn)) % check we didnt already do this one
            shared_sents = [shared_sents, sn];  % noqa
        end
    end 
end
                 
%% generate continous matrices

% params
n_sents = length(out);
n_sua = size(out(1).spikes_sua, 1);
n_mel = size(out(1).aud, 1);
concat_length = 643100;  % overshoot a bit 

% init files
ecog_tc = zeros(256, concat_length);
sua_spikes = zeros(n_sua, concat_length);
mel_spec = zeros(n_mel, concat_length);
phn_tc = zeros(length(phn_cats), concat_length);
feature_tc = zeros(length(phonetic_features)+2, concat_length); % plus sentence and word onsets
formant_tc = zeros(4, concat_length);
formant_normed_tc = zeros(4, concat_length);
word_probs = zeros(3, concat_length);

%% loop through and make res

smin = 1;
shared_sentences_added = [];
for si = 1:n_sents
    
    % we will need this later
    sentence_label = out(si).name;
    
    % only continue if this is one of the shared ones
    if ~any(strcmp(shared_sents, sentence_label))
        continue
        % dont add it more than once
%         if any(strcmp(shared_sentences_added, sentence_label))
%             continue
%         end
    end

    % get sentence data
    this_sua = out(si).spikes_sua;
    
    % get mel data - downsample: this is an error in making the out
    this_mel = out(si).aud;
    
    % determine the size of this sentence
    n_samps = size(this_sua, 2);
    smax = smin + n_samps - 1 - 850;  % reduce to 400ms after sentence
    n_new_samps = smax-smin;
    
    % add data to the preallocation
    sua_spikes(:, smin:smax) = this_sua(:, 1:n_new_samps+1);
    mel_spec(:, smin:smax) = this_mel(:, 1:n_new_samps+1);
    
    % get the ecog
    ec_idx = find(strcmp(ecog_sents, sentence_label));
    this_ecog = ecog(ec_idx).resp;
    
    % average if this is a rep
    if length(size(this_ecog)) == 3
        this_ecog = mean(this_ecog, 3);
    end
    
    % add the phoneme stuff
    for phn_i = 1:length(phn_evnt)
                
        % only add if it is the right sentence 
        if strcmp(sentence_label, phn_evnt(phn_i).sent_name)

            % the npx and mel have already been aligned to the first phoneme
            % not the beginning of the file, so we need to account for that
            % when adding the phonetic stuff
            if phn_evnt(phn_i).sentence_onset == 1
                first_phn_onset = floor(phn_evnt(phn_i).start / 16);
            end

            % we also need to account for it in the ecog too, because this is
            % just to sentence onset not first phoneme onset!
            % so i just need to subtract this from the ecog baseline
            
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
        
    % add it
    ecog_tc(:, smin:smax) = this_ecog(:, first_phn_onset:n_new_samps+first_phn_onset);
        
    % collect the formant info
    if strcmp(sentdet(si).name, out(si).name)
        f = sentdet(si).formants;
        formant_tc(:, smin:smax) = f(:, 1:n_new_samps+1);
        
        % a pitch normalised version
        for form = 1:4
            formant_normed_tc(form, smin:smax) = formant_tc(form, smin:smax) / nanmean(formant_tc(form, smin:smax));
        end
    end
        
    end
        
    % update the smin
    smin = smax;
    
    % how we doin?
    disp(si);
    
end

% normalise the formants
formant_tc = formant_tc / max(max(formant_tc));

%% save it
save_mel_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_mel-1000.mat';
save_sua_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/sua93-1000.mat';
save_fea_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_fea-1000.mat';
save_phn_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_phn-1000.mat';
save_formant_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_formant-1000.mat';
save_formant_normed_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_formant_normed-1000.mat';
%save_ecog_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog-1000.mat';
save_probs_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/probs-1000.mat';

save(save_mel_fname, 'mel_spec', '-v7');
save(save_sua_fname, 'sua_spikes', '-v7');
save(save_phn_fname, 'phn_tc', '-v7');
save(save_fea_fname, 'feature_tc', '-v7');
save(save_formant_fname, 'formant_tc', '-v7');
save(save_formant_normed_fname, 'formant_normed_tc', '-v7');
%save(save_ecog_fname, 'ecog_tc', '-v7');
save(save_probs_fname, 'word_probs', '-v7');
