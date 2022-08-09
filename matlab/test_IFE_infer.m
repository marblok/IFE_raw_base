function test_IFE_infer

do_draw = 1
alfa = 2;

results_folder = '..\test_results\KeelePitch\const_SNR_test_2022_03_23_MLP2\'
wav_folder = '..\test_data\KeelePitch\forValidation\'

filename_core = 'f5nw0000'

wav_filename = [wav_folder, filename_core, '.wav']
F0_filename = [wav_folder, filename_core, '.csv']
F0_IFE_filename = [results_folder, filename_core, '.mat']

[data, fs] = audioread(wav_filename);
F0_ref = readmatrix(F0_filename);
ref_f0_per_sample = F0_ref.';
F0_IFE = load(F0_IFE_filename, '-mat');
F0_IFE = F0_IFE.F0_est;

if do_draw == 1
    timestamps = (1:length(data))./fs;
    my_figure = figure('units','normalized','outerposition',[0 0 1 1]);
    spectrogram(data,kaiser(alfa*256,5),alfa*(256-16),alfa*512,fs, 'yaxis');
    hold on
    set(plot(timestamps, ref_f0_per_sample/1000, 'k'), 'LineWidth', 0.5, 'DisplayName', 'F0_{ref}');
    set(plot(timestamps, F0_IFE/1000, 'r--'), 'LineWidth', 0.5, 'DisplayName', 'F0_{IFE}');
    hold off
%     if do_save == 1
%         saveas(gcf,fullfile(outpath, strcat(filename_prefix, '.png')));
%     end
    set(gca, 'Clim', [-100, -40])

    legend
end
