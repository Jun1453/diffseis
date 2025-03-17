import segyio
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, decimate, resample

x2offsets = lambda x_list: [x[0,:] for x in x_list]

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 4):
    b, a = butter(poles, cutoff, 'highpass', fs=sample_rate)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def lowpass1d(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 4):
    b, a = butter(poles, cutoff, 'lowpass', fs=sample_rate)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def get_xyz(raws, offsets, num_sample: int, sampling_rate=250, reduction_velocity=6.0):
    x = [np.array([offset]*(num_sample))/1000 for offset in offsets]
    y = [(np.array([list(range(num_sample))]*len(offset)).T/sampling_rate) for offset in offsets]
    if reduction_velocity is not None:
        for i in range(len(y)): y[i] -= abs(x[i])/reduction_velocity
    z = [raw.T[:num_sample,:] for raw in raws]
    return x,y,z

def read_segy(filename, fix_unsigned_offset=False, reduction_velocity=None, original_delta=250, downsample_t_every=None):
    # new_delta = original_delta if downsample_t_every is None else original_delta/downsample_t_every
    # downsample_t = lambda arr: [lowpass1d(el[:,::downsample_t_every], new_delta/2.1, new_delta) for el in arr] if downsample_t_every is not None else arr
    # downsample_t = lambda arr: [el[:,::downsample_t_every] for el in arr] if downsample_t_every is not None else arr
    if downsample_t_every is None:
        resample_t = lambda arr: arr
        new_delta = original_delta
    elif downsample_t_every < 0:
        resample_t = lambda arr: [resample(el, int(-downsample_t_every*el.shape[-1]), axis=-1) for el in arr]
        new_delta = int(-downsample_t_every*original_delta)
    else:
        resample_t = lambda arr: [decimate(el, downsample_t_every, ftype='fir') for el in arr] 
        new_delta = original_delta/downsample_t_every
    with segyio.open(filename, ignore_geometry=True) as f:
        headers = f.header

        if np.log10(headers[0][1]) > 5:
            raws = [np.array([f.trace.raw[i] for i in range(len(f.trace.raw))])]
            offsets = [np.array([header[37] for header in f.header])]
        else:
            ntrace_ends = []
            for idx, header in enumerate(headers):
                # when shot number defined by TRACE_SEQUENCE_LINE
                if header[21]==0:
                    if header[1]//10000 > len(ntrace_ends):
                        ntrace_ends.append(idx)
                # when station is defined as FieldRecord
                else:
                    if header[9] > len(ntrace_ends):
                        ntrace_ends.append(idx)
            ntrace_ends.append(None)

            raws = [np.array(f.trace.raw[ntrace_ends[i]:ntrace_ends[i+1]]) for i in range(len(ntrace_ends)-1)]
            offsets = [np.array([header[37] for header in f.header[ntrace_ends[i]:ntrace_ends[i+1]]]) for i in range(len(ntrace_ends)-1)]

        if fix_unsigned_offset:
            for idx in range(len(offsets)):
                if offsets[idx][np.argmin(offsets[idx])-1] < offsets[idx][np.argmin(offsets[idx])+1]:
                    offsets[idx][:np.argmin(offsets[idx])] *= -1
                else:
                    offsets[idx][:np.argmin(offsets[idx])+1] *= -1
        raws = resample_t(raws)
        # print(raws[0].shape)
        num_sample = headers[0][115] if (downsample_t_every is None) or downsample_t_every>0 else int(headers[0][115]*-downsample_t_every)
        return get_xyz(raws, offsets, num_sample=num_sample, reduction_velocity=reduction_velocity, sampling_rate=new_delta)

def calculate_reducuction(z_list, x_list, start_reduction_time=0.5, end_lapse_time=3.06, reduction_velocity=6.0, sampling_rate=250):
    offsets = x2offsets(x_list)
    new_z = np.zeros(int(end_lapse_time*sampling_rate), x_list.shape[1])
    for l in range(len(z_list)):
        timeseries = prefilter(z_list[l]) if prefilter else z_list[l]
        offset_distance = offsets[l]

def output_image(z_list, x_list, vclip, imgdir, image_size=(256,64), start_reduced_time=0.5, prefilter=None, end_lapse_time=3.06, geospread_corr=0, output_npy=False, reduction_velocity=6.0, sampling_rate=250, initial_count=0):
    offsets = x2offsets(x_list)
    crop = np.zeros(shape=image_size)
    count = 0
    for l in range(len(z_list)):
        timeseries = prefilter(z_list[l]) if prefilter else z_list[l]
        offset_distance = offsets[l]
        time_loop = range(0, int(sampling_rate * end_lapse_time)-crop.shape[0], int(crop.shape[0]*0.2))
        # print(list(time_loop))
        offset_loop = range(0, timeseries.shape[1]-crop.shape[1], int(crop.shape[1]*0.2))
        for i in time_loop:
            for j in offset_loop:
                for k in range(crop.shape[1]):
                    start_sample = i + (int(sampling_rate*(start_reduced_time+abs(offset_distance[j+k])/reduction_velocity)) if reduction_velocity is not None else 0)
                    end_sample = start_sample+crop.shape[0]
                    crop[:,k] = timeseries[start_sample:end_sample,j+k]*max(1,geospread_corr*(abs(offset_distance[j+k])/20))

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
    return count

def diversity_stack(z_list, x_list, prefilter=lambda z: highpass(z, 2, 250), first_arrival_reference=None):
    offsets = x2offsets(x_list)
    cube = np.array([(prefilter(z) if prefilter else z) for z in z_list])
    # print(cube.shape)
    noise = np.ones((cube.shape[0],cube.shape[2]))
    for j in range(cube.shape[2]):
        for i in range(cube.shape[0]):
            start = int(250*abs(offsets[i][j])/6)
            if first_arrival_reference is not None:
                end = start+125+int(first_arrival_reference[min(j,len(first_arrival_reference)-1)])-25
            else:
                end = start+250
                
            if (abs(offsets[i][j]) > 6) or ((abs(offsets[i][j]) > 2.65) and (end<start+275+(125*max(np.amin(10+offsets[i])/-40,1)))):
                # noise[i,j] = np.mean(np.abs(cube[i,start:end,j]))
                noise[i,j] = np.sqrt(np.mean(cube[i,start:end,j]**2))
            else:
                # noise[i,j] = np.mean(np.abs(cube[i,:50,j]))
                noise[i,j] = np.sqrt(np.mean(cube[i,:50,j]**2))
            
            # cube[i,:,j] /= noise[i,j]
            cube[i,:,j] *= ((1/noise[i,j]) / max(0.1, np.sum(1/noise[:,j])))
    cube_stacked = np.sum(cube,axis=0)
    return cube_stacked
