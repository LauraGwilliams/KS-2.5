subj = 'NP12';
block = 'B5';

stimTimes = [60 1000]; % NP10_B2: [390 910]; NP10_B3: [230 740]; NP10_B4: [385 900];
%rawdatadir = '/userdata/matt_l/neuropixels';
rawdatadir = '/userdata/lgwilliams/neuropixels/data';

addpath(genpath('/home/matt_l/matlab/SpikeGLX_Datafile_Tools/MATLAB'));
addpath(genpath('/home/matt_l/matlab/KS-2.5'));
% addpath(genpath('/home/matt_l/matlab/continuous_speech'));

binName = [subj '_' block '_g0_t0.nidq.bin'];
binPath = [rawdatadir '/' subj '/raw/' subj '_' block '_g0'];

% Parse the corresponding metafile
meta = ReadMeta(binName, binPath);

% Get first one second of data
nSamp = floor(1.0 * SampRate(meta));

clear dataArray dataArrayA

t0 = round(str2double(meta.niSampRate)*stimTimes(1));
t1 = round(str2double(meta.niSampRate)*(stimTimes(2)-stimTimes(1))); %str2double(meta.fileTimeSecs));
dataArray = ReadBin(t0,t1,meta,binName,binPath);
dataArrayA = GainCorrectNI(dataArray, 1:4, meta);
clear dataArray

%% plot ANIN channel

figure;
plot(dataArrayA(3, 1:600000));

%% create ANIN files

addpath(genpath('/home/matt_l/matlab/prelimAnalysis_matlab'));

if ~exist([binPath '/' subj '_' block '_g0_imec0/Analog'],'dir')
    mkdir([binPath '/' subj '_' block '_g0_imec0/Analog']);
end
for i = 1:size(dataArrayA,1)
    writehtk([binPath '/' subj '_' block '_g0_imec0/Analog/ANIN' num2str(i) '.htk'],...
        dataArrayA(i,:),str2double(meta.niSampRate));
end

% rmpath(genpath('/home/matt_l/matlab/prelimAnalysis_matlab'));

%% find events

% Which TIMIT block is which, in order of how they would be sorted
% when calling ls() -- remember these won't always be in numerical
% order
timit_numbers = [1]; 

% Which analog channel to use for each block (usually you want ANIN2,
% the speaker channel, but sometimes it's bad and you need to use the 
% mic channel instead).
anin_to_use =   [3];

subpath = binPath; %'/Users/mattleonard/Documents/Research/data/raw_data'; %Whereever the subj ECoG Block data is
%subpath = '/crtx3/Leah/segmentedTIMIT/CAR256/';

%subj = input('subject id: ','s'); % only use for interactive version!
subj = upper(subj);
expt = ECoGBlockNames(sprintf('%s/%s',[subpath]));

disp(['Found ' num2str(length(expt)) ' blocks'])
disp(expt)

wpath = '/home/matt_l/matlab/KS-2.5/findEvents/@ECSpeech/Sounds';

fnames = dir([wpath filesep '*.wav']);
all_names = {};
for iAudio = 1:length(fnames)
    all_names{iAudio} = strrep(fnames(iAudio).name,'.wav','');
end

disp('starting to find the events...')

load TIMITopt_fileList.mat;
load stimorder_TIMITopt_withblock5_blocks1_5.mat;
stimorder = stimorder1;
start_file =[1 201]; %[1 101 201 301];
names = {};
for timitblock=1%:3
    names{timitblock} = [];
    names{timitblock}=timitfile(stimorder(start_file(timitblock):start_file(timitblock+1)-1));
end
% for timit5
timit5 = load('TIMIT5_list.mat');
names{3} = timit5.timit5names;

dpath = [subpath];

for exp=1:length(expt)
    evnt(exp).evnt = struct();
end
debug_fig=0;
evnt = DetectEventsQuick(dpath, wpath, expt, timit_numbers, names, all_names, anin_to_use, debug_fig);

[~,idx] = sort([evnt.StartTime]);
evnt = evnt(idx);

spath = [rawdatadir filesep subj filesep 'raw' filesep subj '_' block '_g0' filesep subj '_' block '_g0_imec0' filesep subj '_evnt.mat'];
save(spath,'evnt')
disp(['saving to ' spath])

%% Correct Start Times

T = readtable([rawdatadir filesep subj filesep 'raw' filesep subj '_' block '_g0' filesep subj '_' block '_g0_imec0' filesep subj '_' block '_evnt_CORRECTED.csv']);
for i = 1:length(evnt)
    evnt(i).StartTime = T.CorrectStartTime(idx(i));
    evnt(i).CorrectedVal = T.CorrectedVal(idx(i));
    evnt(i).CorrectStartTime = T.CorrectStartTime(idx(i));
end

spath = [rawdatadir filesep subj filesep 'raw' filesep subj '_' block '_g0' filesep subj '_' block '_g0_imec0' filesep subj '_evnt_corrected.mat'];
save(spath,'evnt');
disp(['saving to ' spath]);
