import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import segyio
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from accelerate import Accelerator
from torch.optim import Adam
from scipy.signal import butter, filtfilt, decimate, resample


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 4):
    b, a = butter(poles, cutoff, 'highpass', fs=sample_rate)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def jamstec_handler(f):
    headers = f.header         
    # deal with TRACE_SEQUENCE_LINE numbered using more than 5 digits
    if np.log10(headers[0][1]) > 5:
        raws = [np.array([f.trace.raw[i] for i in range(len(f.trace.raw))]).T]
        offsets = [np.array([header[37] for header in f.header]).T]
    else:
        ntrace_ends = []
        for idx, header in enumerate(headers):
            # when shot number defined by TRACE_SEQUENCE_LINE
            if header[21]==0:
                if header[1]//10000 > len(ntrace_ends): # when a new shot number appears
                    ntrace_ends.append(idx)
            # when station is defined as FieldRecord
            else:
                if header[9] > len(ntrace_ends):
                    ntrace_ends.append(idx)
        ntrace_ends.append(None)
        raws = [np.array(f.trace.raw[ntrace_ends[i]:ntrace_ends[i+1]]).T for i in range(len(ntrace_ends)-1)]
        offsets = [np.array([header[37] for header in f.header[ntrace_ends[i]:ntrace_ends[i+1]]]).T/1000 for i in range(len(ntrace_ends)-1)]

    # Pad arrays to match longest trace length
    max_len = max(raw.shape[1] for raw in raws)
    padded_raws = []
    padded_offsets = []
    for raw, offset in zip(raws, offsets):
        pad_width = ((0, 0), (0, max_len - raw.shape[1]))
        # print(pad_width)
        padded_raw = np.pad(raw, pad_width, mode='edge')
        padded_raws.append(padded_raw)
        padded_offset = np.pad(offset, pad_width[-1], mode='edge')
        padded_offsets.append(padded_offset)
    return padded_raws, padded_offsets

class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
        self.backup = {}

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def apply_shadow(self):
        """Save current model parameters and replace them with EMA parameters."""
        self.backup = {name: param.data.clone() for name, param in self.module.state_dict().items()}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.module.state_dict()[name])

    def restore(self):
        """Restore the original model parameters."""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

class Profiles(np.ndarray):
    def __new__(cls, input_array, sampling_rate=None, filter_history=None, reduction_vel=None, offsets=None, first_arrival_reference=None):
        obj = np.asarray(input_array, dtype=np.float32).view(cls)
        obj.sampling_rate = sampling_rate  # Sampling rate in Hz
        obj.filter_history = filter_history  # Tuple of (low_freq, high_freq) in Hz
        obj.reduction_vel = reduction_vel  # Reduction velocity in km/s
        # Offsets in km
        if type(offsets) is not list:
            if offsets.ndim == 1:
                offsets = [offsets]
            else:
                offsets = [ offset for offset in offsets]
        obj.offsets = offsets
        # First arrival in sec
        if first_arrival_reference is None:
            obj.first_arrival_reference = [ [None for _ in p] for p in offsets]
        else:
            if type(first_arrival_reference) is not list:
                if first_arrival_reference.ndim == 1:
                    first_arrival_reference = [first_arrival_reference]
                else:
                    first_arrival_reference = [ fa for fa in first_arrival_reference]
            else:
                obj.first_arrival_reference = first_arrival_reference  
        return obj
    
    def __add__(self, other):
        if (type(other) == tuple) or (type(other) == list):
            return type(self).concatenate([self] + list(other))
        else:
            return type(self).concatenate([self, other])

    def __array_finalize__(self, obj):
        if obj is None: return
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.filter_history = getattr(obj, 'filter_history', None) 
        self.reduction_vel = getattr(obj, 'reduction_vel', None)
        self.offsets = getattr(obj, 'offsets', None)
        self.first_arrival_reference = getattr(obj, 'first_arrival_reference', None)

    def __getitem__(self, key):
        # Get the sliced array
        sliced_array = super().__getitem__(key)
        # If the result is a Profiles instance, update its offsets
        if isinstance(sliced_array, type(self)):
            if isinstance(key, tuple):
                if type(key[0]) is slice:
                    # For multi-profile slicing
                    if len(key) == 3:
                        sliced_array.offsets = [offset[key[-1]] for offset in self.offsets[key[0]]]
                        if self.first_arrival_reference is not None:
                            sliced_array.first_arrival_reference = [fa[key[-1]] if fa is not None else None for fa in self.first_arrival_reference[key[0]]]
                    else:
                        sliced_array.offsets = self.offsets[key[-1]]
                        if self.first_arrival_reference is not None: 
                            fa = self.first_arrival_reference
                            sliced_array.first_arrival_reference = fa[key[-1]] if fa is not None else None
                else:
                    sliced_array.offsets = self.offsets[key[0]][key[-1]]
                    if self.first_arrival_reference is not None:
                        try:
                            fa = self.first_arrival_reference[key[0]]
                            sliced_array.first_arrival_reference = fa[key[-1]] if fa is not None else None
                        except:
                            raise ValueError(sliced_array.first_arrival_reference)
            else:
                sliced_array.offsets = self.offsets[key]
                if self.first_arrival_reference is not None:
                    sliced_array.first_arrival_reference = self.first_arrival_reference[key]
        return sliced_array

    def overwrite_segy(self, segy_file): # for Fujie san
        with segyio.open(segy_file, 'r+', strict=False) as f:
            f.raw = self # placeholder, more processing is needed
            f.write()

    @classmethod
    def load(cls, segy_file, data_handle, filter_history=[], reduction_vel=0):
        """Load seismic data from SEGY file into Profiles instance"""
        with segyio.open(segy_file, 'r', strict=False) as f:
            raws, offsets = data_handle(f)
            sampling_rate = 1e6/f.header[0][117]
        
        return cls(raws,
                  sampling_rate=sampling_rate, 
                  filter_history=filter_history,
                  reduction_vel=reduction_vel,
                  offsets=offsets)
    
    @classmethod
    def concatenate(cls, profiles):
        """Concatenate multiple Profile instances along dimension 0.
        
        Args:
            profiles: List/tuple of Profiles instances to concatenate
            
        Returns:
            New Profiles instance with concatenated data and offsets
            
        Raises:
            ValueError: If profiles have different shapes (except dim 0) or sampling rates
        """
        if not profiles:
            raise ValueError("No profiles provided for concatenation")
            
        # Check shapes and sampling rates match
        base_shape = profiles[0].shape[1:]
        base_rate = profiles[0].sampling_rate
        for p in profiles[1:]:
            if p.shape[1:] != base_shape:
                raise ValueError("All profiles must have same shape (except dimension 0)")
            if p.sampling_rate != base_rate:
                raise ValueError("All profiles must have same sampling rate")
                
        # Concatenate data and offsets
        concat_data = np.concatenate(profiles, axis=0)
        concat_offsets = []
        concat_arrivals = []
        for p in profiles:
            concat_offsets += p.offsets
            # if profiles[0].first_arrival_reference:
            concat_arrivals += p.first_arrival_reference if p.first_arrival_reference else [None]
        
        return cls(concat_data,
                  first_arrival_reference=concat_arrivals,# if concat_arrivals else None,
                  sampling_rate=base_rate,
                  filter_history=profiles[0].filter_history,
                  reduction_vel=profiles[0].reduction_vel,
                  offsets=concat_offsets)

    def diversity_stack(self, first_arrival_reference=None, orig_profile_num=False, normalize_to_profile_num=False, normalize_to_original_level=False):
        """Stack along first dimension using diversity stack method"""
        if self.ndim < 2:
            raise ValueError("Array must have at least 2 dimensions")
        
        # # Simple mean stack for now - can be enhanced with true diversity stack
        # stacked = np.mean(self, axis=0)

        # Diversity stack
        stacked = np.zeros_like(self)
        noise = np.ones((self.shape[0],self.shape[2])) * float(np.median(np.sqrt((np.ravel(self[:,:50,:]**2)))))
        for j in range(self.shape[2]):
            for i in range(self.shape[0]):
                start = int(self.sampling_rate*abs(self.offsets[i][j])/self.reduction_vel)
                if first_arrival_reference is not None:
                    end = start+int(first_arrival_reference[min(j,len(first_arrival_reference)-1)])-25
                else:
                    end = start+250
                    
                if (abs(self.offsets[i][j]) > 6) or ((abs(self.offsets[i][j]) > 2.65) and (end<start+275+(125*max(np.amin(10+self.offsets[i])/-40,1)))):
                    # noise[i,j] = np.mean(np.abs(self[i,start:end,j]))
                    noise[i,j] = np.sqrt(np.mean(self[i,start:end,j]**2))
                else:
                    # noise[i,j] = np.mean(np.abs(self[i,:50,j]))
                    # if (abs(self.offsets[i][j]) < 3.8):
                    if False:
                        central_stns = int(8.0/(self.offsets[i][1]-self.offsets[i][0]))
                        if j<central_stns:
                            noise[i,j] = np.sqrt(np.mean(self[i,:50,j+central_stns]**2))
                        else:
                            noise[i,j] = np.sqrt(np.mean(self[i,:50,j-central_stns]**2))
                    else:
                        noise[i,j] = np.sqrt(np.mean(self[i,:50,j]**2))
                        # noise[i,j] = 1.
                        pass
        
                stacked[i,:,j] = self[i,:,j] * (1/noise[i,j])
            # stacked[:,:,j] = stacked[:,:,j] / np.sum(1/noise[:,j])
        # # normalize by noise
        # sorted_noise = np.sort(noise.flatten())
        # bottom_50_percent = sorted_noise[:int(len(sorted_noise) * 0.50)]
        # noise_std = np.std(bottom_50_percent)
        # noise_mean = np.mean(noise)
        # for j in range(self.shape[2]):
        #     for i in range(self.shape[0]):
        #         # stacked[i,:,j] = self[i,:,j] * (1/noise[i,j]) #* (1/stacked.shape[0]) #if (normalize_to_profile_num) or (normalize_to_one) else 1)
        #         # stacked[i,:,j] = self[i,:,j] * ((1/noise[i,j]) / max(0.1, np.sum(1/noise[:,j])))
        #         stacked[i,:,j] = self[i,:,j] * (1/noise[i,j]) / np.maximum(self.shape[0]/noise_mean, np.sum(1/noise[:,j]))
        #         # stacked[i,:,j] = self[i,:,j] * (1/noise[i,j]) / np.sum(1/noise[:,j])
        #         # if normalize_to_original_level: stacked[:,:,j] = stacked[:,:,j] / np.mean(1/noise[:,j]) 
        # # if normalize_to_original_level: stacked[:,:,:] = stacked[:,:,:] / np.median(1/noise[:,:]) 
        # # for j in range(self.shape[2]):
        # #     stacked[:,:,j] = stacked[:,:,j] / np.sum(1/noise[:,j])

        # normalize by signal
        for j in range(self.shape[2]):
            signal_sum = np.sum([np.sqrt(np.mean(stacked[k,:,j]**2)) for k in range(stacked.shape[0])])
            for i in range(self.shape[0]):
                signal_ratio = np.sqrt(np.mean(stacked[i,:,j]**2)) / signal_sum
                stacked[i,:,j] = stacked[i,:,j] * signal_ratio
                if j ==0: print(i, signal_ratio)
        stacked = np.sum(stacked, axis=0)

        # stacked = np.mean(stacked, axis=0)
        if first_arrival_reference is None:
            first_arrival_reference = [first_arrival_reference] * len(self.offsets[0])

        if orig_profile_num:
            return type(self)([stacked]*len(self.offsets),
                            first_arrival_reference=[first_arrival_reference]*len(self.offsets),
                            sampling_rate=self.sampling_rate,
                            filter_history=self.filter_history,
                            reduction_vel=self.reduction_vel,
                            offsets=self.offsets)
        else:
            return type(self)([stacked],
                            first_arrival_reference=[first_arrival_reference],
                            sampling_rate=self.sampling_rate,
                            filter_history=self.filter_history,
                            reduction_vel=self.reduction_vel,
                            offsets=[np.mean(self.offsets, axis=0)])

    def filter(self, filter_func):
        filtered = [np.array([filter_func(self[i,:,j]) for j in range(self.shape[2])]).T for i in range(self.shape[0])]
        return type(self)(filtered,
                         first_arrival_reference=self.first_arrival_reference,
                         sampling_rate=self.sampling_rate,
                         filter_history=self.filter_history + [filter_func],
                         reduction_vel=self.reduction_vel,
                         offsets=self.offsets)
    
    def reduction(self, reduction_vel):
        self.reduction_vel = reduction_vel
        return self

    def write(self, filename):
        """Save Profiles instance to file"""
        np.savez(filename, 
                 data=self,
                 first_arrival_reference=self.first_arrival_reference,
                 sampling_rate=self.sampling_rate,
                 filter_history=self.filter_history,
                 reduction_vel=self.reduction_vel,
                 offsets=self.offsets)
    @classmethod 
    def read(cls, filename, allow_pickle=True):
        """Read Profiles instance from file"""
        npzfile = np.load(filename, allow_pickle=allow_pickle)
        return cls(npzfile['data'],
                  sampling_rate=npzfile['sampling_rate'].item(),
                  filter_history=list(npzfile['filter_history']),
                  reduction_vel=npzfile['reduction_vel'].item(),
                  offsets=npzfile['offsets'],
                  first_arrival_reference=npzfile['first_arrival_reference'])
    
    def get_xyz(self, raw, offset, sample_num=None):
        if sample_num == None:
            sample_num = self.shape[1]
        x = np.array([offset]*(sample_num))
        y = np.array([np.arange(sample_num)]*len(offset)).T/self.sampling_rate
        if self.reduction_vel:
            for j in range(len(y)): y[j] -= abs(x[j])/self.reduction_vel
        z = raw[:sample_num,:]
        return x,y,z
    
    def geospread_corr(self, func=lambda x: max(1, 1*(abs(x)/20))):
        for i in range(self.shape[0]):
            for j in range(self.shape[2]):
                self[i,:,j] *= func(self.offsets[i][j])
        return self

    def reshape_from_2d(self):
        if len(self.shape) == 2:
            self = self.reshape(1, *self.shape)
            self.offsets= [self.offsets]
            self.first_arrival_reference = [self.first_arrival_reference]
        return self

    def plot(self, figsize=None, cmap='seismic', vmin=-1, vmax=1, tmax=None, label_offset=True, plot_reference_arrival=False):
        """Generate subplots for each profile along dimension 0
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        """
        import matplotlib.pyplot as plt

        self = self.reshape_from_2d()
        
        n_profiles = self.shape[0]
        if figsize is None: figsize = (6,0.5+self.shape[0]*1.5) 
        fig, axes = plt.subplots(n_profiles, 1, figsize=figsize, sharex=True)
        if n_profiles == 1:
            axes = [axes]
            
        # time = np.arange(self.shape[1]) / self.sampling_rate
        for i, ax in enumerate(axes):
            if tmax: 
                time_range = tmax + (max(abs(self.offsets[i]))/self.reduction_vel if self.reduction_vel else 0)
                sample_num = int(time_range*self.sampling_rate)
                x,y,z = self.get_xyz(self[i], self.offsets[i], sample_num)
                ax.set_ylim(0,tmax)
            else:
                x,y,z = self.get_xyz(self[i], self.offsets[i])

            # print(x.shape,y.shape,z.shape){}

            if not label_offset:
                x = np.array([np.arange(len(self.offsets[i]))]*sample_num)
            im = ax.pcolorfast(x[:z.shape[0],:z.shape[1]], y[:z.shape[0],:z.shape[1]], z[:-1,:-1], cmap=cmap, vmin=vmin, vmax=vmax)

            if plot_reference_arrival:
                if self.first_arrival_reference is None:
                    raise Exception("First arrival reference is not defined in the Profiles instance.")
                else:
                    if self.first_arrival_reference[i] is not None:
                        # print(self.offsets[i])
                        im = ax.plot(self.offsets[i], self.first_arrival_reference[i]/self.sampling_rate, 'y', label='reference arrival', alpha=0.7)

            if i == len(axes)-1:
                if label_offset: ax.set_xlabel('Offset (km)')
                else: ax.set_xlabel('Channel')
            ax.invert_yaxis()
            ax.set_ylabel('Time (s)')
            # ax.set_xlabel('Channel')
            ax.set_aspect('auto')
            # ax.set_title(f'Profile {i+1}')
            
        plt.tight_layout()
        return fig, axes

    def output_image(self, vclip, imgdir, image_size=(256,64), starttime=0.5, endtime=3.06, geospread_corr=0, output_npy=False, initial_count=0):
        for pfl in range(self.shape[0]):
            count = 0
            crop = np.zeros(shape=image_size)
            timeseries = self[pfl,:,:]
            offset_distance = self.offsets[pfl]
            time_loop = range(0, int(self.sampling_rate*endtime)-crop.shape[0], int(crop.shape[0]*0.2))
            offset_loop = range(0, timeseries.shape[1]-crop.shape[1], int(crop.shape[1]*0.2))
            for i in time_loop:
                for j in offset_loop:
                    for k in range(crop.shape[1]):
                        start_sample = i + (int(self.sampling_rate*(starttime+abs(offset_distance[j+k])/self.reduction_vel)) if self.reduction_vel is not None else 0)
                        end_sample = start_sample+crop.shape[0]
                        crop[:,k] = timeseries[start_sample:end_sample,j+k]*max(1,geospread_corr*(abs(offset_distance[j+k])/20))

                    Path(imgdir).mkdir(parents=True, exist_ok=True)
                    if output_npy:
                        crop = np.float32(crop/vclip)
                        np.save(f'{imgdir}/{initial_count+count}', crop.T)
                    else:
                        crop[crop<-vclip] = -vclip
                        crop[crop>vclip] = vclip
                        crop+=vclip; crop/=(2*vclip);
                        crop_int=np.uint8(255*crop)
                        # crop = np.int8
                        # Image.fromarray(crop_int).save(f'{imgdir}/st{t}rt{i}of{j}.png')
                        Image.fromarray(crop_int.T).save(f'{imgdir}/{initial_count+count}.png')
                    count+=1
            print(f"Image saved: ({count}/{initial_count+count}) units with {len(offset_loop)} on offset axis and {len(time_loop)} on time axis")
            initial_count += count
        return initial_count

    def fragmentize(self, vclip=50, tmin=0.5, tmax=None, t_interval=3.04, unit_size=(64,256), x_move_ratio=0.2, y_move_ratio=0.2):
        """return iterable dataset of 2-d array fragments for pytorch dataloader"""
        self = self.reshape_from_2d()
        if (tmin is None) and (tmax is None) and (t_interval is None):
            time_crop = None
        else:
            if tmin is None: tmin = 0
            if tmax is None: tmax = tmin + t_interval
            time_crop = (tmin, tmax)
        if vclip is None: vclip = 1
        return self.Fragment(self/vclip, unit_size=unit_size, time_crop=time_crop, x_move=int(unit_size[0]*x_move_ratio), y_move=int(unit_size[1]*y_move_ratio))
    
    class Fragment(data.Dataset):
        def __init__(self, profiles, unit_size: tuple, time_crop, x_move: int, y_move: int):
            self.profiles = profiles
            self.unit_size = unit_size
            self.x_move = x_move
            self.y_move = y_move
            self.time_crop = time_crop
            if time_crop is None: sample_min, sample_max = (0, profiles.shape[1])
            else: sample_min, sample_max = (int(time_crop[0]*profiles.sampling_rate), int(time_crop[1]*profiles.sampling_rate))
            self.x_tile = 1 + (profiles.shape[2]-unit_size[0])//x_move
            self.y_tile = 1 + ((sample_max-sample_min)-unit_size[1])//y_move
            self.fragments = np.zeros((self.profiles.shape[0]*self.x_tile*self.y_tile , unit_size[0], unit_size[1]))
            print(self.fragments.shape)

            for i in range(self.fragments.shape[0]):
                num_profile = i // (self.x_tile * self.y_tile)
                num_x_tile = (i % (self.x_tile * self.y_tile)) % self.x_tile
                num_y_tile = (i % (self.x_tile * self.y_tile)) // self.x_tile
                loc_x_start = num_x_tile * x_move
                loc_y_start = num_y_tile * y_move + sample_min
                if self.profiles.reduction_vel:
                    reduced_times_in_sample = np.abs(self.profiles.offsets[num_profile][loc_x_start:loc_x_start+unit_size[0]]
                                                                /self.profiles.reduction_vel)*self.profiles.sampling_rate
                    # print(reduced_times_in_sample)
                    for j in range(unit_size[0]):
                        loc_reduced_y_start = int(loc_y_start + reduced_times_in_sample[j])
                        buffer_start = max(min(unit_size[1], self.profiles.shape[1]-loc_reduced_y_start),0)
                        if buffer_start > 0:
                            self.fragments[i,j,:buffer_start] = self.profiles[num_profile, loc_reduced_y_start:loc_reduced_y_start+unit_size[1], loc_x_start+j]
                        if buffer_start < unit_size[1]:
                            self.fragments[i,j,buffer_start:] = np.zeros((unit_size[1]-buffer_start))
                else:
                    self.fragments[i] = self.profiles[num_profile, loc_x_start:loc_x_start+unit_size[0], loc_y_start:loc_y_start+unit_size[1]]
        
        def __len__(self):
            return self.profiles.shape[0] * self.x_tile * self.y_tile
        
        def __getitem__(self, key):
            if hasattr(self, 'ground_truth'):
                return torch.from_numpy(np.float32(self.fragments[key])).unsqueeze(dim=0), torch.from_numpy(np.float32(self.ground_truth.fragments[key])).unsqueeze(dim=0)
            else:
                return torch.from_numpy(np.float32(self.fragments[key])).unsqueeze(dim=0)
        
        def __setitem__(self, key, value):
            self.fragments[key] = value

        def set_ground_truth(self, ground_truth):
            if len(self) != len(ground_truth): raise ValueError("size of the given ground_truth does not match")
            self.ground_truth = ground_truth
        
        def rebuild(self, x_move_factor=None, y_move_factor=None, x_move=None, y_move=None) -> "Profiles":
            if x_move_factor is not None:
                rebuild_x_move = int(self.x_move * x_move_factor)
            elif x_move is not None:
                rebuild_x_move = x_move
            else:
                rebuild_x_move = self.x_move

            if x_move_factor is not None:
                rebuild_y_move = int(self.y_move * y_move_factor)
            elif y_move is not None:
                rebuild_y_move = y_move
            else:
                rebuild_y_move = self.y_move
            
            rebuilt_x_size = (self.unit_size[0]+self.x_move*(self.x_tile-1))
            rebuilt_y_size = (self.unit_size[1]+self.y_move*(self.y_tile-1))
            rebuilt_profiles = type(self.profiles)(np.zeros((self.profiles.shape[0], rebuilt_y_size, rebuilt_x_size)),
                                                    first_arrival_reference=self.profiles.first_arrival_reference,
                                                    sampling_rate=self.profiles.sampling_rate,
                                                    filter_history=self.profiles.filter_history,
                                                    reduction_vel=0.,
                                                    offsets=self.profiles.offsets)
            
            for i in range(self.profiles.shape[0]):
                weight = np.zeros((rebuilt_y_size, rebuilt_x_size))
                for j in list(range(0, self.y_tile-1, rebuild_y_move//self.y_move))+[self.y_tile-1]:
                    loc_y_start = j * self.y_move
                    loc_y_end = loc_y_start+self.unit_size[1]
                    for k in list(range(0, self.x_tile-1, rebuild_x_move//self.x_move))+[self.x_tile-1]:
                        loc_x_start = k * self.x_move
                        loc_x_end = loc_x_start+self.unit_size[0]

                        index = k + j*self.x_tile + i*self.x_tile*self.y_tile
                        rebuilt_profiles[i, loc_y_start:loc_y_end, loc_x_start:loc_x_end] += self[index].numpy()[0].T
                        weight[loc_y_start:loc_y_end, loc_x_start:loc_x_end] += np.ones((self.unit_size[1], self.unit_size[0]))
                    
            
            return rebuilt_profiles/weight

        def denoise(self, ddpm, parameter_dir, batch_size=32, device='cuda'):
            if hasattr(self, 'ground_truth'): raise Exception('Fragment with appointed target data cannot be denoised')
            parameters = torch.load(parameter_dir, map_location=torch.device(device), weights_only=True)['model']

            del parameters['betas']
            del parameters['alphas_cumprod']
            del parameters['alphas_cumprod_prev']
            del parameters['sqrt_alphas_cumprod']
            del parameters['sqrt_one_minus_alphas_cumprod']
            del parameters['log_one_minus_alphas_cumprod']
            del parameters['sqrt_recip_alphas_cumprod']
            del parameters['sqrt_recipm1_alphas_cumprod']
            del parameters['posterior_variance']
            del parameters['posterior_log_variance_clipped']
            del parameters['posterior_mean_coef1']
            del parameters['posterior_mean_coef2']


            def change_key(self, old, new):
                #copy = self.copy()
                for _ in range(len(self)):
                    k, v = self.popitem(False)
                    self[new if old == k else k] = v
                    
            keys = []
            for key, value in parameters.items():
                keys.append(key)
                
            for i in range(len(keys)):
                change_key(parameters, keys[i], keys[i][11:])
                
            ddpm.denoise_fn.load_state_dict(parameters)
            dl = data.DataLoader(self, batch_size=batch_size, pin_memory=True)


            count = 0
            results = type(self)(profiles=self.profiles,
                                    unit_size=self.unit_size,
                                    time_crop=self.time_crop,
                                    x_move=self.x_move,
                                    y_move=self.y_move)
            for input_data in dl:
                output_data = ddpm.inference(x_in=input_data.to(device), clip_denoised=False)
                for i in range(len(input_data),2*len(input_data)):
                    results[count] = output_data[i,0].cpu().detach()
                    count += 1
            # [ddpm.inference(x_in=input_data.to(device), clip_denoised=False)[i,0].cpu().detach().numpy() for input_data in dl for i in range(len(input_data),2*len(input_data))]


            return results
        
        def train(self, ddpm, num_epochs, batch_size=32, learning_rate=3e-6, enable_amp=True, pre_ema_epoch=5, ema_decay=0.995, gradient_accumulate_every=2, save_every=None, results_folder='.', load_from=None):
            if not hasattr(self, 'ground_truth'): raise Exception('Model cannot be trained with Fragment with no appointed target data')
            if save_every is None: save_every = num_epochs

            # Initialize accelerator
            accelerator = Accelerator(mixed_precision='fp16' if enable_amp else 'no')
            device = accelerator.device

            optimizer = Adam(ddpm.parameters(), lr=learning_rate)
            ema = ModelEmaV2(ddpm, ema_decay, device)

            # Initialize model with loaded data
            if load_from is not None:
                load_model = torch.load(str(load_from), map_location=device)
                load_epoch = load_model['epoch']
                ddpm.load_state_dict(load_model['model'])
                ema.load_state_dict(load_model['ema'])
                optimizer.load_state_dict(load_model['optimizer'])
            else:
                load_epoch = 0

            # Prepare for distributed training
            ddpm, optimizer = accelerator.prepare(ddpm, optimizer)
            dl = data.DataLoader(self, shuffle=True, batch_size=batch_size, pin_memory=True)
            dl = accelerator.prepare(dl)

            for n in range(load_epoch+1, num_epochs+1):
                total_loss = 0
                count = 0
                if accelerator.is_main_process:
                    loop = tqdm(dl, total=len(dl)*(num_epochs-load_epoch), desc=f"Training DDPM @ Epoch {n}", initial=len(dl)*(n-1-load_epoch))
                else: loop = dl
                
                for d, gt in loop:
                    count += 1
                    
                    # Forward pass
                    loss = ddpm(d, gt)
                    loss = loss / gradient_accumulate_every

                    # Check for NaN loss
                    if torch.isnan(loss):
                        if accelerator.is_main_process:
                            tqdm.write('Warning: NaN loss encountered, skipping update.')
                        count -= 1
                        continue
                    
                    # Backward pass
                    accelerator.backward(loss)

                    if count % gradient_accumulate_every == 0:
                        total_loss += loss.item() * gradient_accumulate_every
                        if accelerator.is_main_process:
                            tqdm.write(f'Loss: {loss.item():.4e} -> Avg Loss: {(total_loss/count):.4e}')
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        if n > pre_ema_epoch: ema.update(ddpm)

                if accelerator.is_main_process:
                    print(f'Epoch {n}: {(total_loss/count):.4e}')

                    # if n > pre_ema_epoch:
                    #     ema.apply_shadow()
                    #     ema.restore()

                    if (n % save_every == 0) or (n == num_epochs):
                        milestone = n // save_every if n < num_epochs else 'final'
                        info = {
                            'epoch': n,
                            # 'model': accelerator.unwrap_model(ddpm).state_dict(),
                            'model': accelerator.unwrap_model(ema.module).state_dict(),
                            'ema': ema.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }
                        accelerator.save(info, str(Path(results_folder) / f'model-{milestone}.pt'))

            if accelerator.is_main_process:
                print('training completed')

        def validate(self, ddpm, batch_size=32, enable_amp=True, gradient_accumulate_every=2, load_from=None, torch_device='cuda'):
            if not hasattr(self, 'ground_truth'): raise Exception('Model cannot be evaluated with Fragment with no appointed target data')
            if load_from is None: raise Exception('Model cannot be evaluated without specified parameters to load')

            # Initialize accelerator
            accelerator = Accelerator(mixed_precision='fp16' if enable_amp else 'no')
            device = accelerator.device

            # Initialize model with loaded data
            load_model = torch.load(str(load_from), map_location=device)
            load_epoch = load_model['epoch']
            ddpm.load_state_dict(load_model['model'])

            # Prepare for distributed training
            dl = data.DataLoader(self, shuffle=True, batch_size=batch_size, pin_memory=True)
            ddpm, dl = accelerator.prepare(ddpm, dl)

            for n in range(load_epoch, load_epoch+1):
                total_loss = 0
                count = 0
                if accelerator.is_main_process:
                    loop = tqdm(dl, total=len(dl), desc=f"Evaluating DDPM @ Epoch {n}")
                else: loop = dl
                
                for d, gt in loop:
                    count += 1
                    
                    # Forward pass
                    loss = ddpm(d, gt)
                    loss = loss / gradient_accumulate_every

                    # Check for NaN loss
                    if torch.isnan(loss):
                        if accelerator.is_main_process:
                            tqdm.write('Warning: NaN loss encountered, skipping validation.')
                        count -= 1
                        continue

                    if count % gradient_accumulate_every == 0:
                        total_loss += loss.item() * gradient_accumulate_every
                        # if accelerator.is_main_process:
                        #     tqdm.write(f'Loss: {loss.item():.4e} -> Avg Loss: {(total_loss/count):.4e}')

                if accelerator.is_main_process:
                    print(f'Epoch {n}: {(total_loss/count):.4e}')

            if accelerator.is_main_process:
                print('validation completed')
