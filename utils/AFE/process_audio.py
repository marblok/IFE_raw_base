import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from struct import pack
from tqdm import tqdm

from scipy.io.wavfile import read as audioread
from scipy.io import savemat as save
from scipy.io import loadmat
from scipy.signal import lfilter, freqz, spectrogram, convolve
from scipy.signal import welch as pwelch
from torch._C import dtype

class FB_params_class:
    def __init__(self, F_s = 8000, F_0_min = 50, F_k_max = 1050, K = 20, q = 2.0, 
        filter_bank_mode = 'ConstQ', filter_bank_smoothing_mode= 'delay', CMPO_smoothing_mode='none',
        FB_to_CMPO_smoothing_factor = 1.0):

        self.F_s = F_s # sampling rate
        self.F_0_min = F_0_min # first filter center frequency
        self.F_k_max = F_k_max # last filter center frequency
        self.K = K # number of filters
        self.q = q # filters "quality factor"
        self.filter_bank_mode = filter_bank_mode
        self.filter_bank_smoothing_mode = filter_bank_smoothing_mode
        self.CMPO_smoothing_mode = CMPO_smoothing_mode
        self.FB_to_CMPO_smoothing_factor = FB_to_CMPO_smoothing_factor

        self.F_k = np.zeros(self.K)
        self.F_k[:] = np.nan
        self.N_k = np.zeros(self.K)
        self.N_k[:] = np.nan
        self.delay = np.zeros(self.K)
        self.max_delay = 0
        # self.h_k = [np.zeros((0,1)) for _ in range(self.K)]
        self.h_k = None
        self.h_delay_k = None
        self.h_CMPO_smoothing_k = None
        
    def load(self, filename):
        # logging.info(data)
        # data = loadmat(filename, struct_as_record=False)
        data = loadmat(filename)

        #print(data["CQT"])
        CQT_data = data["CQT"].dtype
        CQT = data["CQT"]
        
        for fn in CQT_data.names:
            #print(f"fn:{fn}")
            val = CQT[fn]
            if val.shape == (1,1):
                val = val.item((0,0))
                if val.shape == (1,1):
                    val = val.item((0,0))
                elif val.shape == (1,):
                    val = val
                else:
                    # sh = val.shape
                    tmp = list()
                    if val.size > len(val):
                        for v_ in val[0]:
                            if v_.size > 1:
                                tmp.append(v_[0])
                            else:
                                tmp.append(v_)
                        val = tmp
                    else:
                        val = val.tolist()
                    #len_2  = len(val)
                setattr(self, fn, val)
            else:
                print(f"val:{val}")
                print(f"shape:{val.shape}")
                raise Exception("unexpected val.shape")

        # self.F_s = CQT.F_s[0,0]
        # self.F_0_min = CQT.F_0_min[0,0]
        # self.F_k_max = CQT.F_k_max[0,0]
        # self.K = CQT.K[0,0]
        # self.q = CQT.q[0,0]
        # self.mode = CQT.filter_bank_mode[0]


def process_audio(filename_in = None, x  = None, no_of_filters = 20, AFE_cfg = None, Fs=None, do_draw=False, process_audio_version = 2):
    if (filename_in is None) and (x is None):
        filename_in = '.\input_data\audio_files\3000events_50-400Hz_length500ms_gaps0ms.wav'

    ## filterbank params
    # Przykłady parametrów dla ConstQ

    if AFE_cfg is None:
        AFE_cfg = {"filter_bank_mode": "ConstQ", 
                   "filter_bank_smoothing_mode": "delay",
                   "CMPO_smoothing_mode": "none",
                   "FB_to_CMPO_smoothing_factor": 1.0} # full FB smoothing/delay
        raise Exception("please provide AFE_cfg param")
    else:
        if "FB_to_CMPO_smoothing_factor" not in AFE_cfg:
            AFE_cfg["FB_to_CMPO_smoothing_factor"] = 1.0
            logging.info(f"WARNING: AFE configuration without FB_to_CMPO_smoothing_factor")

    if "q" not in AFE_cfg:
        AFE_cfg["q"] = 2.0
    FB_params = FB_params_class(F_s=8000, F_0_min=50, F_k_max=1050, K=no_of_filters, 
        # q=2.0, # mode='ConstQ',
        q=AFE_cfg["q"], # mode='ConstQ',
        filter_bank_mode=AFE_cfg["filter_bank_mode"], 
        filter_bank_smoothing_mode=AFE_cfg["filter_bank_smoothing_mode"],
        CMPO_smoothing_mode=AFE_cfg["CMPO_smoothing_mode"],
        FB_to_CMPO_smoothing_factor=AFE_cfg["FB_to_CMPO_smoothing_factor"])
    # %FB_params = struct('F_s', 8000, 'F_0_min', 50, 'F_k_max', 2100, 'K', 100, 'q', 2, 'mode', 'ConstQ');
    # % FB_params = struct('F_s', 8000, 'F_0_min', 50, 'F_k_max', 2100, 'K', 200, 'q', 2, 'mode', 'ConstQ');

    # % Przykłady parametrów dla ConstBW 
    # % FB_params = struct('F_s', 8000, 'F_0_min', 50, 'F_k_max', 400, 'K', 35, 'q', 2, 'mode', 'ConstBW');
    # % FB_params = struct('F_s', 8000, 'F_0_min', 50, 'F_k_max', 2100, 'K', 100, 'q', 2, 'mode', 'ConstBW');
    # % FB_params = struct('F_s', 8000, 'F_0_min', 50, 'F_k_max', 2100, 'K', 200, 'q', 2, 'mode', 'ConstBW');

    if filename_in is not None:
        if FB_params.filter_bank_mode == 'ConstQ':
            FB_params.filename = f'CQFB_F_min={FB_params.F_0_min}_FB_smooth={FB_params.filter_bank_smoothing_mode}_CMPO_smooth={FB_params.CMPO_smoothing_mode}_FB2CMPO_sr={FB_params.FB_to_CMPO_smoothing_factor:.2f}_F_max={FB_params.F_k_max}_K={FB_params.K}_q={FB_params.q:.2f}_Fs={FB_params.F_s}.mat'

            filename_out = filename_in[:-4] + '_CQFB_K_' + str(FB_params.K) + '.dat'
            # filename_out2 = [filename_in2(1:end-4), '_CQFB_K_', num2str(FB_params.K), '.dat'];

        elif FB_params.filter_bank_mode == 'ConstBW':
            FB_params.filename = f'CBWFB_F_min={FB_params.F_0_min}_FB_smooth={FB_params.filter_bank_smoothing_mode}_CMPO_smooth={FB_params.CMPO_smoothing_mode}_FB2CMPO_sr={FB_params.FB_to_CMPO_smoothing_factor:.2f}_F_max={FB_params.F_k_max}_K={FB_params.K}_q={FB_params.q:.2f}_Fs={FB_params.F_s}.mat'

            filename_out = filename_in[:-4] + '_CBWFB_K_' + str(FB_params.K) + '.dat'
            #%filename_out2 = [filename_in2(1:end-4), '_CBWFB_K_', num2str(FB_params.K), '.dat'];
        
        else:
            raise Exception(f"unsupported FB_params.filter_bank_mode={FB_params.filter_bank_mode}")
    else:
        FB_params.filename = None
        filename_out = None

    # TODO test filename_out

    logging.info(f"using process_audio_version: {process_audio_version}")
    # %% file processing
    if process_audio_version == 1:
        if filename_in is None:
            [FB_params, FB_CMPO, FB_data] = process_audio_data(signal=x, Fs=Fs, CQT=FB_params, do_draw=do_draw)
        else:
            [FB_params, FB_CMPO, FB_data] = process_audio_data(filename_in=filename_in, filename_out=filename_out, CQT=FB_params, do_draw=do_draw)
    if process_audio_version == 2: 
        if filename_in is None:
            [FB_params, FB_data] = process_audio_data_2(signal=x, Fs=Fs, CQT=FB_params)
        else:
            # [FB_params, FB_data] = process_audio_data_2(filename_in=filename_in, filename_out=filename_out, CQT=FB_params)
            [FB_params, FB_data] = process_audio_data_2(filename_in=filename_in, filename_out=None, CQT=FB_params)
        FB_CMPO = None
            
    # CQT.Bins_per_octave
    if do_draw == True:
        draw_CQT_CMPO(FB_params, FB_CMPO, 1)

    # % 
    # %[FB_params, FB_CMPO] = process_audio_data(filename_in2, filename_out2, FB_params, false);
    # %draw_CQT_CMPO(FB_params, FB_CMPO, 2)

    ## function draw_CQT_CMPO
    return FB_params, FB_data, FB_CMPO


def process_audio_data(CQT, signal=None, Fs=8000, filename_in=None, filename_out=None, do_draw = False):
    # CQT - either designed CQT filter bank or design parameters

    if signal is not None:
        x = signal
        F_s = Fs
    else:
        # filename_in = '..\sound_model\test_shimmer.wav';
        [F_s, x] = audioread(filename_in)
        if x.dtype == np.dtype(np.int16):
            logging.info("read 16-bit int wave file")
            amplitude = np.iinfo(np.int16).max
            x = x.astype('float')/amplitude
        elif x.dtype == np.dtype(np.int32):
            logging.info("read 24/32-bit int wave file")
            amplitude = np.iinfo(np.int32).max
            x = x.astype('float')/amplitude
        elif x.dtype == np.dtype(np.uint8):
            logging.info("read 8-bit uint wave file")
            x = (x.astype('float')-128)/128
        elif x.dtype == np.dtype(np.float32):
            logging.info("read 32-bit float wave file")
        else:
            logging.error(f"read unsupported wave file, dtype={x.dtype}")

    # x = randn(size(x));

    # % M = F_s/8000;
    # % 
    # % if M - floor(M) > 0
    # %   error('F_s has to be a multiple of 8000 Sa/s')
    # % end
    # % if M > 1
    # %   warning('No a.a. filter is applied')
    # %   pause
    # % end
    # % 
    # % x = x(1:M:end);
    # % F_s = F_s/M;

    if F_s != 8000:
        logging.warning('Input file has to be resampled !!!')
        input("Press Enter")
    
        # TODO import change_sample_rate from m-file
        x = change_sample_rate(x, F_s, 8000)
        F_s = 8000

    # % % x = randn(size(x));
    # % n = 0:length(x);
    # % F_0 = 30;
    # % 
    # % x = sin(2*pi*F_0/F_s*n);
    # % 
    # % % x = chirp(n/F_s,F_0,n(end)/F_s,4000);
    # % % % spectrogram(x,kaiser(256,5),220,512,F_s, 'yaxis');
    # % 
    # % % pause

    # % % zakres oczekiwanych częstotliwości tonu krtaniowego (F_0)
    # % F_0_min = 30; 
    # % F_k_max = F_s/4;
    # % 
    # % % H_s_max = floor(0.5*F_s/F_0_min) % dwie próbki na okres dla minimalnej F0
    # % % H_s_min = floor(0.5*F_s/F_k_max) % dwie próbki na okres dla maksymalnej F0
    # % % UWAGA: dla F_s = 8000 oraz F_k_max = 2000 => H_s_min = 2
    # % %  - o ile możliwy jest pomiar jitteru, to w zasadzie dla tego zakresu
    # % %    oczekiwanych F0 nie bardzo ma sens stosowanie "hop size" przy pomiarze
    # % 
    # % % UWAGA: ???? jeżeli w uczeniu nie wykorzystujemy "pamięci", to dla danych
    # % %    treningowych nie trzeba wykorzystywać wszystkch kolejnych "ramek"
    # % %    => problemem mogą jednak być fluktuacje pulsacji wyższych harmonicznych !!!
    # % 
    # % Bins_per_octave = 3; % number of bins per octave
    # % q = 2; % współczynnik proporcjonalności dobroci filtów Q

    # % design if necessary and use CQT filter bank
    # %[CMPO_CQT, CQT] = compute_CQT(x, CQT, do_draw);
    [CMPO_CQT, CQT] = compute_CQT(x, CQT, False)
    N = len(CMPO_CQT[0])
    frames = np.zeros((2*CQT.K, N), dtype=np.float32) # kolejne ramki/zestawy w kolejnych kolumnach
    frames[:] = np.nan

    if do_draw == True:
        fig = plt.figure(5)
        axs =fig.subplots(2)
        ha1 = axs[0]
        ha2 = axs[1]

        spec_K_factor = 4
        [S,F,T] = spectrogram(x,window=np.blackman(spec_K_factor*256),noverlap=spec_K_factor*240,nfft=spec_K_factor*512,fs=F_s) #,'yaxis')
        # hi = image(ha2, T, F, 10*log10(abs(S)))
        hi = ha2.imshow(T, F, 10*np.log10(np.abs(S)))
        set(hi, 'CDataMapping', 'scaled')
        set(ha2, 'Ydir', 'normal')
        # hold(ha2, 'on')
        
        max_P = 0

    #P_inst = np.empty(shape=(CQT.K,1))
    P_inst = [None] * CQT.K
    #F_inst = np.empty(shape=(CQT.K,1))
    F_inst = [None] * CQT.K
    for k in range(CQT.K):  # 1:CQT.K
        # %   power_factor = 2*CQT.Q(end) * 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
        #   power_factor = 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
        power_factor = 1; # normalizacja do amplitudy sinusoidy
        
        # TODO These are probably vectors not scalars
        P_inst[k] = power_factor*np.abs(CMPO_CQT[k]); 
        F_inst[k] = np.angle(CMPO_CQT[k], deg=False)/(2*np.pi)*F_s

        frames[2*k, :] = P_inst[k]
        frames[2*k+1, :] = F_inst[k]
        
        if do_draw == True:
            max_P = max([max_P, max(P_inst[k])])
        
            ha1.plot(F_inst[k], P_inst[k])
            # hold(ha1, 'on')

            #     plotc(ha2, (0:length(F_inst)-1)/F_s, F_inst, P_inst);

    if do_draw == True:
        for k in range(CQT.K-1,-1,-1):  # CQT.K:-1:1
            #     hp = plot(ha2, (0:length(F_inst)-1)/F_s, F_inst);
            #     set(hp, 'LineWidth', 2*max(P_inst)/max_P);
            F_inst_tmp = F_inst[k]
            F_inst_tmp[P_inst[k] < max(P_inst[k])*0.05] = np.nan
            F_inst_tmp[P_inst[k] < max_P*0.001] = np.nan
            P_inst_tmp = P_inst[k]
            t = np.arange(0,len(F_inst_tmp))/F_s

            while len(F_inst_tmp) > 0:
                #array[numpy.isfinite(array)][0]
                #np.argwhere(np.isnan(x))
                #ind1 = find(~isnan(F_inst_tmp), 1); # pierwszy nie nan
                ind1 = np.argwhere(~np.isnan(F_inst_tmp))[0] # pierwszy nie nan
                F_inst_tmp = F_inst_tmp[ind1:], P_inst_tmp = P_inst_tmp[ind1:], t = t[ind1:]
                #ind2 = find(isnan(F_inst_tmp), 1); % pierwszy nan
                ind2 = np.argwhere(np.isnan(F_inst_tmp))[0] # pierwszy nan
                if len(ind2) == 0:
                    ind2 = len(F_inst_tmp)
                F_inst_tmp_ = F_inst_tmp[ind1:ind2]
                P_inst_tmp_ = P_inst_tmp[ind1:ind2]
                t_ = t[ind1:ind2+1]
                #      hp = plot(ha2, t_, F_inst_tmp_, 'Color', sqrt(1-max(P_inst_tmp_)/max_P)*[1, 1, 1]);
                hp = ha2.plot(t_, F_inst_tmp_, 'Color', np.power((1-max(P_inst_tmp_)/max(P_inst[k])),(1/10))*[1, 1, 0])

                F_inst_tmp[ind1:ind2] = []
                P_inst_tmp[ind1:ind2] = []
                t[ind1:ind2] = []

            #     hp = plot(ha2, t, F_inst_tmp, 'Color', (1-max(P_inst)/max_P)*[1, 1, 1]);
    
        K = 8
        [Pxx,F] = pwelch(x,window=np.kaiser(K*256,5),noverlap=K*220,nfft=K*1024,fs=F_s, return_onesided=True)
        Pxx = max_P * Pxx/max(Pxx)
        ha1.plot(F, Pxx, 'k')

        # hold(ha1, 'off')
        # hold(ha2, 'off')

    if filename_out is not None:
        logging.info(f"Saving: {filename_out}")
        with open(filename_out, 'wb') as hf:
            header_data = [CQT.K, 2, N, CQT.F_s, CQT.F_0_min, CQT.F_k_max, CQT.Bins_per_octave, CQT.q]
            # 255 #  'short' little endian # mark version 2 header
            hf.write(pack('<4h', 255, len(header_data), *header_data[0:2])) #  'short' little endian
            hf.write(pack('<Q', header_data[2])) # 'unsigned long long'
            l_ = len(header_data[3:])
            hf.write(pack('<{}d'.format(l_), *[float(x) for x in header_data[3:]])) # 'double'
            # list_of_frames = np.transpose(frames).tolist()
            # print(list_of_frames[:5])
            # hf.write(pack('<f', *list_of_frames)) # 'float'
            #np.transpose(frames).tofile(hf, '', 'float32')
            np.transpose(frames).astype('float32').tofile(hf)
        logging.info(f"Finished saving: {filename_out}")

        # write to csv
        #filename_out = [filename_out(1:end-4), '.csv'];
        #writematrix(frames',filename_out);

    return CQT, CMPO_CQT, np.transpose(frames)

def process_audio_data_2(CQT, signal=None, Fs=8000, filename_in=None, filename_out=None):
    # CQT - either designed CQT filter bank or design parameters

    if signal is not None:
        x = signal
        F_s = Fs
    else:
        # filename_in = '..\sound_model\test_shimmer.wav';
        [F_s, x] = audioread(filename_in)
        if x.dtype == np.dtype(np.int16):
            logging.info("read 16-bit int wave file")
            amplitude = np.iinfo(np.int16).max
            x = x.astype('float')/amplitude
        elif x.dtype == np.dtype(np.int32):
            logging.info("read 24/32-bit int wave file")
            amplitude = np.iinfo(np.int32).max
            x = x.astype('float')/amplitude
        elif x.dtype == np.dtype(np.uint8):
            logging.info("read 8-bit uint wave file")
            x = (x.astype('float')-128)/128
        elif x.dtype == np.dtype(np.float32):
            logging.info("read 32-bit float wave file")
        else:
            logging.error(f"read unsupported wave file, dtype={x.dtype}")

    if F_s != 8000:
        logging.warning('Input file has to be resampled !!!')
        input("Press Enter")
    
        # TODO import change_sample_rate from m-file
        x = change_sample_rate(x, F_s, 8000)
        F_s = 8000


    # % design if necessary and use CQT filter bank
    # %[CMPO_CQT, CQT] = compute_CQT(x, CQT, do_draw);
    CQT = compute_CQT_2(CQT)
    # TODO jeżeli filtry CQT miałyby opóźnienie ułamkowe 1/2
    # to kompensowałyby one opóźnienie 1/2 z CMPO
    CMPO_delay = CQT.max_delay+0.5
    
    N = len(x)
    x = np.append(x, np.zeros(shape=(int(CMPO_delay)+1, 1), dtype=np.float32)) # +1 for CMPO buffer
    
    # \TODO transponować frames tak żeby już transponowane wypełniać i nie robić tego później
    frames = np.zeros((2*CQT.K, N), dtype=np.float32) # kolejne ramki/zestawy w kolejnych kolumnach
    frames[:] = np.nan

    with tqdm(total=2*CQT.K, dynamic_ncols=True) as progressbar:
        for k in range(CQT.K):  # 1:CQT.K

            #tmp = CQT.delayed_h_k[k]
            # filter bank filter
            if hasattr(CQT, "delayed_h_k") == False:
                y_CQT_k = lfilter(CQT.h_k[k], 1, x).astype(np.complex64)
            else:
                # for the compatibility with the older CQT format
                if len(CQT.delayed_h_k[k]) > 0:
                    y_CQT_k = lfilter(CQT.delayed_h_k[k], 1, x).astype(np.complex64)
            progressbar.update(1)

            # smoothing / delay compensation
            if hasattr(CQT, "h_delay_k"):
                if len(CQT.h_delay_k[k]) > 0:
                    y_CQT_k = lfilter(CQT.h_delay_k[k], 1, y_CQT_k).astype(np.complex64)
            # kompensacja opóźnienia + jedną próbkę "zgubi" CMPO
            y_CQT_k =  y_CQT_k[int(CMPO_delay):]
            
            CMPO_CQT_k = y_CQT_k[1:] * np.conj(y_CQT_k[0:-1])

            if hasattr(CQT, "h_CMPO_smoothing_k"):
                if len(CQT.h_CMPO_smoothing_k[k]) > 0:
                    CMPO_CQT_k = lfilter(CQT.h_CMPO_smoothing_k[k], 1, CMPO_CQT_k).astype(np.complex64)

            # progressbar.update(1)

            # %   power_factor = 2*CQT.Q(end) * 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
            #   power_factor = 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
            power_factor = 1; # normalizacja do amplitudy sinusoidy
            
            # # These are probably vectors not scalars
            # P_inst[k] = power_factor*np.abs(CMPO_CQT_k); 
            # F_inst[k] = np.angle(CMPO_CQT_k, deg=False)/(2*np.pi)*F_s
            # frames[2*k, :] = P_inst[k]
            # frames[2*k+1, :] = F_inst[k]

            frames[2*k, :] = power_factor*np.abs(CMPO_CQT_k);                 #  P_inst[k]
            frames[2*k+1, :] = np.angle(CMPO_CQT_k, deg=False)/(2*np.pi)*F_s  # F_inst[k]

            progressbar.update(1)


    if filename_out is not None:
        logging.info(f"Saving: {filename_out}")
        with open(filename_out, 'wb') as hf:
            header_data = [CQT.K, 2, N, CQT.F_s, CQT.F_0_min, CQT.F_k_max, CQT.Bins_per_octave, CQT.q]
            # 255 #  'short' little endian # mark version 2 header
            hf.write(pack('<4h', 255, len(header_data), *header_data[0:2])) #  'short' little endian
            hf.write(pack('<Q', header_data[2])) # 'unsigned long long'
            l_ = len(header_data[3:])
            hf.write(pack('<{}d'.format(l_), *[float(x) for x in header_data[3:]])) # 'double'
            # list_of_frames = np.transpose(frames).tolist()
            # print(list_of_frames[:5])
            # hf.write(pack('<f', *list_of_frames)) # 'float'
            #np.transpose(frames).tofile(hf, '', 'float32')
            np.transpose(frames).astype('float32').tofile(hf)
        logging.info(f"Finished saving: {filename_out}")

        # write to csv
        #filename_out = [filename_out(1:end-4), '.csv'];
        #writematrix(frames',filename_out);

    return CQT, np.transpose(frames)

def compute_CQT_2(CQT = None):
    # function [CMPO_CQT, CQT] = compute_CQT(x, F_s, F_0_min, F_k_max, K, q, do_draw)
    if CQT is None:
        CQT = FB_params_class()
        CQT.F_s = 8000

        CQT.F_0_min = 30
        CQT.F_k_max = CQT.F_s/4
        CQT.K = 20 # number of filters in bank
        #   Bins_per_octave = 3; % number of bins per octave

        #   q = 3;
        #   % q = 2.5;
        CQT.q = 2.0
        CQT.filter_bank_mode = 'ConstQ'
        #   CQT.filter_bank_mode = 'ConstBW'

        n = np.arange(0,3000+1)

        F_0 = 97
        x = np.sin(2*np.pi*F_0/CQT.F_s*n, dtype=np.float32)
        x = x + 0.5*np.sin(4*np.pi*F_0/CQT.F_s*n, dtype=np.float32)
        
        do_draw = True

    # number of bins per octave
    CQT.Bins_per_octave = (CQT.K-1)/np.log2(CQT.F_k_max/CQT.F_0_min)
    
    do_not_design_CQT = True
    if CQT.filename is None:
        do_not_design_CQT = False
    else:
        if Path(CQT.filename).is_file():
            # data = load(CQT.filename)
            CQT.load(CQT.filename)
            # logging.info(data)

    # do_not_design_CQT = False;
    if CQT.h_k is None:
        do_not_design_CQT = False

    if do_not_design_CQT == False:
        #   [CQT, K] = get_CQT_filters(CQT.Bins_per_octave, CQT.F_0_min, CQT.F_k_max, CQT.q, CQT.F_s, true);
        if CQT.filter_bank_mode == 'ConstQ':
            CQT = get_CQT_filters(CQT, True)
        else:
            CQT = get_CBW_filters(CQT, True)
        CQT.h_CMPO_smoothing_k = get_CMPO_smoothing_filters(CQT)
        if CQT.filename is not None:
            save(CQT.filename, {"CQT": CQT}, do_compression=True)
        # TODO compare loaded with original CQT structure
    # pause

    return CQT

def draw_CQT_CMPO(CQT, CMPO_CQT, ind):
    fig = plt.figure(1)
    axs = fig.subplots(2)
    gca = axs[ind-1]

    max_P = 0

    for k in range(CQT.K):
        # % %   power_factor = 2*CQT.Q(end) * 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
        # %   power_factor = 1/(CQT.F_k[k]/F_s); % normalizacja do mocy szumu
        power_factor = 1; # normalizacja do amplitudy sinusoidy
    
        P_inst = power_factor*np.abs(CMPO_CQT[k]); 
        F_inst = np.angle(CMPO_CQT[k], deg=False)/(2*np.pi)*CQT.F_s

        max_P = max([max_P, max(P_inst)])
    
        gca.plot(F_inst, P_inst)
        #hold on
        #%   legends_[k] = sprintf('k=%i', k);
    
    
    #hold off
    gca.set_title('Instantaneous Power as a function of instantaneous frequency')
    fig.show()
    # legend(legends_)


def compute_CQT(x, CQT = None, do_draw = False):
    # function [CMPO_CQT, CQT] = compute_CQT(x, F_s, F_0_min, F_k_max, K, q, do_draw)
    if CQT is None:
        CQT = FB_params_class()
        CQT.F_s = 8000

        CQT.F_0_min = 30
        CQT.F_k_max = CQT.F_s/4
        CQT.K = 20 # number of filters in bank
        #   Bins_per_octave = 3; % number of bins per octave

        #   q = 3;
        #   % q = 2.5;
        CQT.q = 2.0
        CQT.filter_bank_mode = 'ConstQ'
        #   CQT.filter_bank_mode = 'ConstBW'

        n = np.arange(0,3000+1)

        F_0 = 97
        x = np.sin(2*np.pi*F_0/CQT.F_s*n, dtype=np.float32)
        x = x + 0.5*np.sin(4*np.pi*F_0/CQT.F_s*n, dtype=np.float32)
        
        do_draw = True

    # number of bins per octave
    CQT.Bins_per_octave = (CQT.K-1)/np.log2(CQT.F_k_max/CQT.F_0_min)

    # % Pytania:
    # % 1. Czy do DNN podawać wyjścia synchroniczne z kompensacja opóźnienia,
    # %    czy też od razu podać wyjścia, tak że parametry z wyższych pasm
    # %    trafiają do DNN wcześniej a z niższych pasm właściwych do parametru sygnału
    # %    później? Synchronizacja strumieni danych spada wtedy na DNN.
    # % 2. Czy można przygotować filtry IIR o mniejszej złożoności oraz mniejszym
    # %    opóźnieniu, z wystarczająco dpbrą charakterystyką opóźnieniową?
    # %    Dodatkowy problem to długosć stanu przejściowego.
    # % 3. Dla zazębiających się pasm różnice amplitud też zachowują informację o
    # %    pulsacji składowej sinusoidalnej z danego pasma powiązaną z
    # %    charakterystyką amplitudową filtru. Czy można o wykorzystać (a) wprost
    # %    wstępnie przetwarzając amplitudy chwilowe z sąsiednich kanałów lub (b)
    # %    zrzucić to na DNN (czy jest w stanie to wykorzystać?)
    # %
    # %    => 3b. eksperyment: można próbować trenować sieć na podstawie samych
    # %    amplitud chwilowych z wyść filtrów!



    N = len(x) # signal length
    do_not_design_CQT = True
    if CQT.filename is None:
        do_not_design_CQT = False
    else:
        if Path(CQT.filename).is_file():
            # data = load(CQT.filename)
            CQT.load(CQT.filename)
            # logging.info(data)

    # do_not_design_CQT = False;
    if CQT.h_k is None:
        do_not_design_CQT = False

    if do_not_design_CQT == False:
        #   [CQT, K] = get_CQT_filters(CQT.Bins_per_octave, CQT.F_0_min, CQT.F_k_max, CQT.q, CQT.F_s, true);
        if CQT.filter_bank_mode == 'ConstQ':
            CQT = get_CQT_filters(CQT, True)
        else:
            CQT = get_CBW_filters(CQT, True)
        if CQT.filename is not None:
            save(CQT.filename, {"CQT": CQT}, do_compression=True)
        # TODO compare loaded with original CQT structure
    # pause

    CMPO_delay = CQT.max_delay+0.5

    # % % measure Q
    # % K_freqz = 8192*32;
    # % for k = 1:CQT.K
    # %   H = freqz(CQT.h_k[k], 1, K_freqz);
    # %   CQT.max_MA[k] = max(abs(H));
    # %   MA = abs(H) / CQT.max_MA[k];
    # %   BW_norm = length(find(MA >= 0.5))/ K_freqz;
    # %   CQT.Q[k] = CQT.F_k[k]/(BW_norm*CQT.F_s);
    # % end

    # TODO jeżeli filtry CQT miałyby opóźnienie ułamkowe 1/2
    # to kompensowałyby one opóźnienie 1/2 z CMPO
    if do_draw == True:
        fig = plt.figure(1)
        
    x_seg = np.append(x, np.zeros(shape=(int(CMPO_delay)+1, 1), dtype=np.float32)) # +1 for CMPO buffer
    # TODO check x_seg size

    y_CQT = list()
    CMPO_CQT = list()

    with tqdm(total=2*CQT.K, dynamic_ncols=True) as progressbar:
        for k in range(CQT.K): # 1:CQT.K
            if do_draw == True:
                logging.info(f"k={k}")
            #tmp = CQT.delayed_h_k[k]
            y_tmp = lfilter(CQT.delayed_h_k[k], 1, x_seg).astype(np.complex64)
            progressbar.update(1)

            #y_CQT.insert(k, y_tmp)
            y_CQT.append(y_tmp)
            # kompensacja opóźnienia + jedną próbkę "zgubi" CMPO
            y_CQT[k] = y_CQT[k][int(CMPO_delay):]
            
            CMPO_CQT.append(y_CQT[k][1:] * np.conj(y_CQT[k][0:-1]))
            progressbar.update(1)

            if do_draw == True:
                axs = fig.subplots(3)
                axs[0].plot(np.abs(CMPO_CQT[k]))
                axs[1].plot(np.angle(CMPO_CQT[k], deg=False))
            #   if abs(CMPO_CQT[k](end) > 0.001
                if any(abs(CMPO_CQT[k]) > 0.00001):
                    axs[2].plot(np.angle(CMPO_CQT[k], deg=False)/(2*np.pi)*CQT.F_s)
                fig.show()
        
    if do_draw == True:
        # subplot(3,1,1)
        # hold off
        # subplot(3,1,2)
        # hold off
        # subplot(3,1,3)
        # hold off

        K = 2
        fig = plt.figure(2)
        axs = fig.subplots(2)
        pwelch(x,window=np.kaiser(K*256,5),noverlap=K*220,nfft=K*1024,fs=CQT.F_s, return_onesided=False)

    return CMPO_CQT, CQT


def get_CQT_filters(CQT, use_half_sample_delay):
    # K = B*log2(F_k_max/F_0_min)+1;% number of frequency bins

    # CQT.F_s = F_s;
    # CQT.K = floor(K);
    CQT.max_delay = 0
    CQT.h_k = []
    for k in range(CQT.K): # 1:CQT.K
        # number of bins per octave
        CQT.Bins_per_octave = (CQT.K-1)/np.log2(CQT.F_k_max/CQT.F_0_min)

        F_k = CQT.F_0_min*np.power(2,(k/CQT.Bins_per_octave))
        N_k = round(CQT.q*CQT.F_s/(F_k*(np.power(2,(1/CQT.Bins_per_octave))-1)))
    
        if use_half_sample_delay == True:
            N_k = N_k + np.remainder(N_k, 2)

            n = np.arange(-(N_k-1)/2,(N_k-1)/2+1)
        else:
            N_k = N_k + (1-np.remainder(N_k, 2))

            n = np.arange(-(N_k-1)/2,(N_k-1)/2+1)

        # TODO: dla okien kosinusowych można wykorzystać wzór okna i zastosować N_k niecałkowite
        w = np.blackman(N_k).astype(np.float32)
        # %   w = hanning(N_k);
        # %   w = 1/N_k * w; 
        w = w / np.sum(w)

        #   a_k = 1/N_k * w(:).*exp(-j*2*pi*n(:)*F_k/F_s);
        h_k = w * np.exp(1j*2*np.pi*n*F_k/CQT.F_s, dtype=np.complex64)

        CQT.F_k[k] = F_k
        CQT.N_k[k] = N_k
        CQT.delay[k] = (N_k-1)/2
        # CQT.h_k[k] = h_k
        CQT.h_k.insert(k, h_k)
        
        if CQT.max_delay < CQT.delay[k]:
            CQT.max_delay = CQT.delay[k]

    # K_freqz = 8*8192
    # H_all = 0
    # H2_all = 0
    # CQT.delayed_h_k = []
    CQT.h_delay_k = [] # delay/smoothing filter
    for k in range(CQT.K): #1:CQT.K
        # % TODO
        # % 1. korekta opóźnienia, tak żeby wszystkei filtry miały takie samo opóźnienie
        # %   a) na potrzeby obliczania H_all
        
        if CQT.max_delay-CQT.delay[k] > 0:
            delay = np.floor(CQT.FB_to_CMPO_smoothing_factor * (CQT.max_delay-CQT.delay[k]))
            if hasattr(CQT, 'filter_bank_smoothing_mode') == False:
                logging.info(f"using default filter_bank_smoothing_mode: delay")
                CQT.filter_bank_smoothing_mode = 'delay'

            is_filter_bank_smoothing_mode_ok = False
            if CQT.filter_bank_smoothing_mode == 'delay':
                # h_k = np.append(np.zeros((int(np.round(delay)), 1), dtype=np.complex64), CQT.h_k[k])
                h_delay_k = np.zeros(int(np.round(delay) + 1))
                h_delay_k[-1] = 1.0
                is_filter_bank_smoothing_mode_ok = True
            if CQT.filter_bank_smoothing_mode == 'rect':
                # h_k = convolve(np.ones((int(np.round(2*delay+1)), 1)/(2*delay+1), dtype=np.float32), CQT.h_k[k])
                N_delay = np.round(2*delay+1)
                h_delay_k = np.ones(int(N_delay))/N_delay
                is_filter_bank_smoothing_mode_ok = True
            if CQT.filter_bank_smoothing_mode == 'none':
                h_delay_k = []
                is_filter_bank_smoothing_mode_ok = True
            if is_filter_bank_smoothing_mode_ok == False:
                raise Exception(f"unexpected is_filter_bank_smoothing_mode:{CQT.filter_bank_smoothing_mode}")
            
        else:
            # h_k = CQT.h_k[k]
            h_delay_k = []
            
        # CQT.delayed_h_k[k] = h_k
        # \TODO separate h_k and smoothing/delay filter
        # CQT.delayed_h_k.insert(k, h_k)
        CQT.h_delay_k.insert(k, h_delay_k)
        # [H, F] = freqz(h_k, 1, K_freqz, whole = True, fs=CQT.F_s)
        # H_all = H_all + H
        # H2_all = H2_all + np.square(np.abs(H))
        
        # n = -(CQT.N_k[k]-1)/2:(CQT.N_k[k]-1)/2
        # %   figure(2)
        # %   subplot(2,1,1)
        # %   plot(n, real(CQT.h_k[k]), 'b')
        # %   hold on
        # %   plot(n, imag(CQT.h_k[k]), 'r')
        # %   hold off
        # %   set(gca, 'Xlim', [-2000,2000])
        # % 
        # %   subplot(2,1,2)
        # %   plot(F, abs(H), 'b');
        # %   plot(F, abs(H).^2, 'r');
        # %   hold on
        
        #pause(0)

    # % plot(F, abs(H_all), 'k');
    # % set(plot(F, H2_all, 'k-'), 'LineWidth', 2);
    # % hold off
    # % 
    # % figure(1)
    # % subplot(3,1,1)
    # % plot(CQT.F_k, 'bo')
    # % subplot(3,1,2)
    # % plot(CQT.N_k, 'bo')
    # % 

    dF_k_min = CQT.F_k[1]-CQT.F_k[0]
    logging.info(f"dF_k_min.K={dF_k_min}")
    logging.info(f"CQT.K={CQT.K}")
    logging.info(f"CQT.max_delay={CQT.max_delay}")

    return CQT

def get_CMPO_smoothing_filters(CQT):

    smoothing_h_k = []
    for k in range(CQT.K): # 1:CQT.K
        delay = CQT.max_delay-CQT.delay[k]
        delay = delay - np.floor(CQT.FB_to_CMPO_smoothing_factor * (CQT.max_delay-CQT.delay[k]))

        if delay > 0:
            if hasattr(CQT, 'CMPO_smoothing_mode') == False:
                logging.info(f"using default CMPO_smoothing_mode: delay")
                CQT.filter_bank_smoothing_mode = 'delay'

            is_CMPO_smoothing_mode_ok = False
            if CQT.CMPO_smoothing_mode == 'delay':
                h_k = np.zeros(int(np.round(delay) + 1))
                h_k[-1] = 1.0
                is_CMPO_smoothing_mode_ok = True
            if CQT.CMPO_smoothing_mode == 'rect':
                N_delay = np.round(2*delay+1)
                h_k = np.ones(int(N_delay))/N_delay
                is_CMPO_smoothing_mode_ok = True
            if CQT.CMPO_smoothing_mode == 'none':
                h_k = []
                is_CMPO_smoothing_mode_ok = True
            if is_CMPO_smoothing_mode_ok == False:
                raise Exception(f"unexpected is_CMPO_smoothing_mode_ok:{CQT.CMPO_smoothing_mode}")
            
        else:
            h_k = []
            
        smoothing_h_k.insert(k, h_k)

    return smoothing_h_k

def get_CBW_filters(CBWFB, use_half_sample_delay):
    # % K = B*log2(F_k_max/F_0_min)+1;% number of frequency bins
    # % N_k = round(CBWFB.q*CBWFB.F_s/(CBWFB.F_0_min*(2^(1/CBWFB.Bins_per_octave)-1)));
    # % N_k = round(CBWFB.q*CBWFB.F_s/CBWFB.F_0_min);

    CBWFB.Bins_per_octave = -1

    CBWFB.max_delay = 0
    dF = (CBWFB.F_k_max - CBWFB.F_0_min) / (CBWFB.K-1)
    N_k = round(CBWFB.q*CBWFB.F_s/dF)
    for k in range(CBWFB.K): # 1:CBWFB.K
        F_k = CBWFB.F_0_min + (k-1)*dF
    
        if use_half_sample_delay == True:
            N_k = N_k + np.remainder(N_k, 2)

            n = np.arange(-(N_k-1)/2,(N_k-1)/2+1)
        else:
            N_k = N_k + (1-np.remainder(N_k, 2))

            n = np.arange(-(N_k-1)/2,(N_k-1)/2+1)

        # TODO: dla okien kosinusowych można wykorzystać wzór okna i zastosować N_k niecałkowite
        w = np.blackman(N_k)
        # %   w = hanning(N_k);
        # %   w = 1/N_k * w; 
        w = w / np.sum(w)

        #   a_k = 1/N_k * w(:).*exp(-j*2*pi*n(:)*F_k/F_s);
        h_k = w*np.exp(1j*2*np.pi*n*F_k/CBWFB.F_s)

        CBWFB.F_k[k] = F_k
        CBWFB.N_k[k] = N_k
        CBWFB.delay[k] = (N_k-1)/2
        CBWFB.h_k[k] = h_k
        
        if CBWFB.max_delay < CBWFB.delay[k]:
            CBWFB.max_delay = CBWFB.delay[k]

    K_freqz = 8*8192
    H_all = 0
    for k in range(CBWFB.K): # 1:CBWFB.K
        # % TODO
        # % 1. korekta opóźnienia, tak żeby wszystkei filtry miały takie samo opóźnienie
        # %   a) na potrzeby obliczania H_all
        
        if CBWFB.max_delay-CBWFB.delay[k] > 0:
            delay = CBWFB.max_delay-CBWFB.delay[k]
            h_k = np.append(np.zeros(int(delay), 1), CBWFB.h_k[k])
        else:
            h_k = CBWFB.h_k[k]
            
        CBWFB.delayed_h_k[k] = h_k
        [H, F] = freqz(h_k, 1, K_freqz, whole = True, fs=CBWFB.F_s)
        H_all = H_all + H
        H2_all = H2_all + np.square(np.abs(H))
        
        # n = np.arange(-(CBWFB.N_k[k]-1)/2,(CBWFB.N_k[k]-1)/2+1)
        # %   figure(2)
        # %   subplot(2,1,1)
        # %   plot(n, real(CBWFB.h_k[k]), 'b')
        # %   hold on
        # %   plot(n, imag(CBWFB.h_k[k]), 'r')
        # %   hold off
        # %   set(gca, 'Xlim', [-2000,2000])
        # % 
        # %   subplot(2,1,2)
        # %   plot(F, abs(H), 'b');
        # %   plot(F, abs(H).^2, 'r');
        # %   hold on
        
        # pause(0)

    # % plot(F, abs(H_all), 'k');
    # % set(plot(F, H2_all, 'k-'), 'LineWidth', 2);
    # % hold off
    # % 
    # % figure(1)
    # % subplot(3,1,1)
    # % plot(CBWFB.F_k, 'bo')
    # % subplot(3,1,2)
    # % plot(CBWFB.N_k, 'bo')


    dF_k_min = CBWFB.F_k(2)-CBWFB.F_k(1)
    logging.info(f"dF_k_min={dF_k_min}")
    logging.info(f"CBWFB.K={CBWFB.K}")
    logging.info(f"CBWFB.max_delay={CBWFB.max_delay}")

    return CBWFB


