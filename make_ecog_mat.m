%% make ecog out
addpath(genpath('/home/lgwilliams/scripts/'));

%% paths
data_path = '/data_store1/human/prcsd_data/EC237/';
save_out = '/userdata/lgwilliams/neuropixels/data/NP04/ecog/';
blocks = {'EC237_B1','EC237_B11','EC237_B20','EC237_B27'};
data_types = {'HilbAA_70to150_8band','RawHTK_1000'};

%% get matrices

for block = blocks
    
    % init
    d_hg = [];
    d_lfp = [];
    i = 1;

    for wav = 1:4

        % get folders
        hg_folder = char(strcat(data_path, block, '/HilbAA_70to150_8band/'));
        lfp_folder = char(strcat(data_path, block, '/RawHTK_1000/'));

        for ch = 1:64

            % file names
            fname_hg = strcat(hg_folder, 'Wav', num2str(wav), num2str(ch), '.htk');
            fname_lfp = strcat(lfp_folder, 'Wav', num2str(wav), num2str(ch), '.htk');

            % load and put
            [tmp, fs] = readhtk([fname_hg]);
            d_hg(i, :) = mean(tmp, 1);
            [tmp, fs] = readhtk([fname_lfp]);
            d_lfp(i,:) = mean(tmp, 1);
            i = i+1;
              
        end
    end
    
    % save out
    hg_save_fname = char(strcat(save_out, block, 'HG.mat'));
    save(hg_save_fname, 'd_hg');
    lfp_save_fname = char(strcat(save_out, block, 'LFP.mat'));
    save(lfp_save_fname, 'd_lfp');
    
end