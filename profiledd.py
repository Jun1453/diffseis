import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import segyio
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
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

class Profiles(np.ndarray):
    def __new__(cls, input_array, sampling_rate=None, filter_history=None, reduction_vel=None, offsets=None, first_arrival_reference=None):
        obj = np.asarray(input_array).view(cls)
        obj.sampling_rate = sampling_rate  # Sampling rate in Hz
        obj.filter_history = filter_history  # Tuple of (low_freq, high_freq) in Hz
        obj.reduction_vel = reduction_vel  # Reduction velocity in km/s
        obj.offsets = offsets  # Offsets in km
        obj.first_arrival_reference = first_arrival_reference  # Offsets in km
        return obj

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
                # For multi-dimensional slicing, only use first dimension for offsets
                sliced_array.offsets = self.offsets[key[0]]
                if self.first_arrival_reference is not None:
                    sliced_array.first_arrival_reference = self.first_arrival_reference[key[0]]
            else:
                sliced_array.offsets = self.offsets[key]
                if self.first_arrival_reference is not None:
                    sliced_array.first_arrival_reference = self.first_arrival_reference[key]
        return sliced_array

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
            concat_arrivals += p.first_arrival_reference
        
        return cls(concat_data,
                  first_arrival_reference=concat_arrivals,
                  sampling_rate=base_rate,
                  filter_history=profiles[0].filter_history,
                  reduction_vel=profiles[0].reduction_vel,
                  offsets=concat_offsets)

    def diversity_stack(self, first_arrival_reference=None, orig_profile_num=False):
        """Stack along first dimension using diversity stack method"""
        if self.ndim < 2:
            raise ValueError("Array must have at least 2 dimensions")
        
        # # Simple mean stack for now - can be enhanced with true diversity stack
        # stacked = np.mean(self, axis=0)

        # Diversity stack
        stacked = np.zeros_like(self)
        noise = np.ones((self.shape[0],self.shape[2]))
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
                    noise[i,j] = np.sqrt(np.mean(self[i,:50,j]**2))
                
                stacked[i,:,j] = self[i,:,j] * (1/noise[i,j])
                # stacked[i,:,j] = self[i,:,j] * ((1/noise[i,j]) / max(0.1, np.sum(1/noise[:,j])))
        stacked = np.sum(stacked, axis=0)

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
    def read(cls, filename):
        """Read Profiles instance from file"""
        npzfile = np.load(filename)
        return cls(npzfile['data'],
                  sampling_rate=npzfile['sampling_rate'].item(),
                  filter_history=list(npzfile['filter_history']),
                  reduction_vel=npzfile['reduction_vel'].item())
    
    def get_xyz(self, raw, offset, sample_num):
        x = np.array([offset]*(sample_num))
        y = np.array([np.arange(sample_num)]*len(offset)).T/self.sampling_rate
        if self.reduction_vel:
            for j in range(len(y)): y[j] -= abs(x[j])/self.reduction_vel
        z = raw[:sample_num,:]
        return x,y,z

    def plot(self, figsize=None, cmap='seismic', vmin=-1, vmax=1, tmax=None, label_offset=True, plot_reference_arrival=False):
        """Generate subplots for each profile along dimension 0
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        """
        import matplotlib.pyplot as plt
        
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
                ax.set_ylim(0,tmax)
            else: sample_num = self.shape[1]

            x,y,z = self.get_xyz(self[i], self.offsets[i], sample_num)
            print(x.shape,y.shape,z.shape)

            if not label_offset:
                x = np.array([np.arange(len(self.offsets[i]))]*sample_num)
            im = ax.pcolorfast(x, y, z[:-1,:-1], cmap=cmap, vmin=vmin, vmax=vmax)

            if plot_reference_arrival:
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
