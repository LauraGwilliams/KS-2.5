% save matlab out struct as a continuous matrix for easy loading into
% python

%% define path to out structure
out_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_preproc_out_1000.mat';
rootZ = '/userdata/lgwilliams/neuropixels/data/NP04/raw/NP04_B2_g0/NP04_B2_g0_imec0';
ecog_fname = '/userdata/lgwilliams/neuropixels/data/NP04/raw/ecog/EC237_TIMIT_HilbAA_70to150_8band_mel_out_resp.mat';

% load ecog
load(ecog_fname);

% get the phonetic info
phn_evnt = table2struct(readtable(fullfile(rootZ, 'EC237_phon_events.csv')));

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
phonetic_features = ["word_onset"; "phonation_v"; "manner_o"; "manner_a"; ...
                     'manner_v'; 'manner_f'; 'manner_n'; "place_l"; ...
                     "place_v"; "place_h"; "place_c"; "place_g"; ...
                     "place_lo"; "place_m"; "place_d"; "frontback_f"; ...
                     "frontback_b";"frontback_n"; "roundness_u"; ...
                     "roundness_r"; "centrality_n"; "centrality_f"; ...
                     "centrality_c"; "surp"; "entropy"];

%% 

feature_labels = ["0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"; "10"; "11"; "12";
       "13"; "14"; "15"; "16"; "17"; "18"; "19"; "20"; "21"; "22"; "23";
       "24"; "25"; "26"; "27"; "28"; "29"; "30"; "31"; "32"; "33"; "34";
       "35"; "36"; "37"; "38"; "39"; "40"; "41"; "42"; "43"; "44"; "45";
       "46"; "47"; "48"; "49"; "50"; "51"; "52"; "53"; "54"; "55"; "56";
       "57"; "58"; "59"; "60"; "61"; "62"; "63"; "64"; "65"; "66"; "67";
       "68"; "69"; "70"; "71"; "72"; "73"; "74"; "75"; "76"; "77"; "78";
       "79"; "F1"; "F2"; "F3"; "F4"; "word_onset"; "phonation_v"; "manner_o"; "manner_a"; ...
       "manner_v"; "manner_f"; "manner_n"; "place_l"; ...
       "place_v"; "place_h"; "place_c"; "place_g"; ...
       "place_lo"; "place_m"; "place_d"; "frontback_f"; ...
       "frontback_b";"frontback_n"; "roundness_u"; ...
       "roundness_r"; "centrality_n"; "centrality_f"; ...
       "centrality_c"; "surp"; "entropy"; "sentence_onset"; "word_onset"; ...
       "unigram"; "trigram"; "gpt2"];

                 
%% generate continous matrices

% params
n_sents = length(out);
concat_length = 1136228;  % overshoot a bit 
n_mel = 80;

% init files
ecog_tc = zeros(256, concat_length);
mel_spec = zeros(n_mel, concat_length);
phn_tc = zeros(length(phn_cats), concat_length);
feature_tc = zeros(length(phonetic_features)+2, concat_length); % plus sentence and word onsets
formant_tc = zeros(4, concat_length);
formant_normed_tc = zeros(4, concat_length);
word_probs = zeros(3, concat_length);

%% loop through and make res

smin = 1;
for si = 1:n_sents
    
    % we will need this later
    sentence_label = out(si).name;

    % get ecog data
    this_ecog = out(si).resp;
    
    % average if this is a rep
    if length(size(this_ecog)) == 3
        this_ecog = mean(this_ecog, 3);
    end
    
    % get mel data
    this_mel = out(si).aud;
    
    % determine the size of this sentence
    n_samps = size(this_ecog, 2);
    smax = smin + n_samps - 1;
    
    % add data to the preallocation
    ecog_tc(:, smin:smax) = this_ecog;
    mel_spec(:, smin:smax) = this_mel(:, 1:smax-smin+1);
    
    % add the phoneme stuff
    for phn_i = 1:length(phn_evnt)
        
        % the npx and mel have already been aligned to the first phoneme
        % not the beginning of the file, so we need to account for that
        %if phn_evnt(phn_i).sentence_onset == 1
        %    first_phn_onset = floor(phn_evnt(phn_i).start / 16);
        %end
        
        % not for ecog
        first_phn_onset = 0;
        
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
        
%     % collect the formant info
%     if strcmp(sentdet(si).name, out(si).name)
%         f = sentdet(si).formants;
%         formant_tc(:, smin:smax) = f(:, 1:n_samps);
%         
%         % a pitch normalised version
%         for form = 1:4
%             formant_normed_tc(form, smin:smax) = formant_tc(form, smin:smax) / nanmean(formant_tc(form, smin:smax));
%         end
%     end
        
    end
        
    % update the smin
    smin = smax;
    
    % how we doin?
    disp(si);
    
    disp(smax);
    
end

%% normalise the formants
formant_tc = formant_tc / max(max(formant_tc));

% save it
save_mel_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog_mel-1000.mat';
save_fea_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog_fea-1000.mat';
save_phn_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog_phn-1000.mat';
% save_formant_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_formant-1000.mat';
% save_formant_normed_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ks_formant_normed-1000.mat';
save_ecog_fname1 = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog-1000-pt1.mat';
save_ecog_fname2 = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog-1000-pt2.mat';
save_probs_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/ecog_probs-1000.mat';

% ecog split
idx = ceil(concat_length/2);
ecog_tc_1 = ecog_tc(:, 1:idx);
ecog_tc_2 = ecog_tc(:, idx+1:end);

% save
save(save_mel_fname, 'mel_spec', '-v7');
save(save_phn_fname, 'phn_tc', '-v7');
save(save_fea_fname, 'feature_tc', '-v7');
% save(save_formant_fname, 'formant_tc', '-v7');
% save(save_formant_normed_fname, 'formant_normed_tc', '-v7');
save(save_ecog_fname1, 'ecog_tc_1', '-v7');
save(save_ecog_fname2, 'ecog_tc_2', '-v7');

save(save_probs_fname, 'word_probs', '-v7');