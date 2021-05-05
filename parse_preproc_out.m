% save matlab out struct as a continuous matrix for easy loading into
% python

%% define path to out structure
out_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/mkl_preproc_out_3000.mat';

% load the out
load(out_fname);

%% generate continous matrices

% params
n_sents = length(out);
n_sua = size(out(1).spikes_sua, 1);
n_mel = size(out(1).aud, 1);
concat_length = 1931000;  % overshoot a bit 

% init files
sua_spikes = zeros(n_sua, concat_length);
mel_spec = zeros(n_mel, concat_length);
smin = 1;
for si = 1:n_sents
    
    % get sentence data
    this_sua = out(si).spikes_sua;
    
    % get mel data - downsample: this is an error in making the out
    this_mel = out(si).aud;
    ratio = size(this_mel, 2) / size(this_sua, 2);
    this_mel = resample(this_mel', 1, 9)';
    
    % determine the size of this sentence
    n_samps = size(this_sua, 2);
    smax = smin + n_samps - 1;
    
    % add data to the preallocation
    sua_spikes(:, smin:smax) = this_sua;
    mel_spec(:, smin:smax) = this_mel(:, 1:smax-smin+1);
    
    % update the smin
    smin = smax;
    
    % how we doin?
    disp(si);
end

%% save it
save_mel_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/mkl_mel.mat';
save_sua_fname = '/userdata/lgwilliams/neuropixels/data/NP04/out_structs/mkl_sua.mat';
save(save_mel_fname, 'mel_spec', '-v7');
save(save_sua_fname, 'sua_spikes', '-v7');