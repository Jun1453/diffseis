import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from obspy.signal.trigger import pk_baer, aic_simple

stn_num_to_n = {
    '01': 100, '02': 15, '03': 145, '04': -15, '05': 0,
    '06': -10, '07': 5, '08': 155, '09': -25, '10': 20,
    '11': 10, '12': 115, '13': 65, '14': 50, '15': 85,
    '16': 90, '17': 45, '18': 150, '19': 30, '20': 135,
    '21': -35, '22': 106, '23': 55, '24': -20, '25': 125, 
    '26': -30, '27': -0.01, '28': -5, '29': 130, '30': 110,
    '31': 80, '32': 95, '33': 120, '34': 140, '35': 75,
    '36': 40, '37': 35, '38': 70, '39': 25, '40': 60 
}

def first_arrival_curve(data):
    p_picks = []
    for i in range(data.shape[1]):
        p_picks.append(aic_simple(data[:1750,i]).argmin())
    # axs[0,0].plot(np.arange(data.shape[1]), p_picks, 'b', label='org out', alpha=0.2)

    # Calculate moving window stats and filter outliers
    window_size = 20
    filtered_p_picks = np.array(p_picks, dtype=float)  # Convert to float to allow None values
    for i in range(len(p_picks)):
        # Get window indices accounting for edges
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(p_picks), i + window_size//2)
        window = p_picks[start_idx:end_idx]
        window.remove(p_picks[i])
        
        # Calculate window statistics
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        # Replace outliers with None
        # if (window_std > 100) or (abs(p_picks[i] - window_mean) > 2 * window_std):
        if (abs(p_picks[i] - window_mean) > 2 * window_std):
            filtered_p_picks[i] = window_mean

    filtered2_p_picks = np.array(p_picks, dtype=float) 
    for i in range(len(filtered_p_picks)):
        # Get window indices accounting for edges
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(filtered_p_picks), i + window_size//2)
        window = filtered_p_picks[start_idx:end_idx]
        # window.remove(filtered_p_picks[i])
        
        # Calculate window statistics
        window_mean = np.mean(window)
        window_std = np.std(window)
        # print(window_std)
        
        # Replace outliers with None
        if (window_std > 150):
            filtered2_p_picks[i] = None
        elif (abs(p_picks[i] - window_mean) > 1 * window_std):
            filtered2_p_picks[i] = window_mean
            
    return p_picks, filtered2_p_picks

def get_frame(canvas_out, canvas_gt):

    # # Get data for current frame
    # frame = frames[frame_idx]
    # canvas_out = frame['canvas_out']
    # canvas_gt = frame['canvas_gt']
    
    # if frame_idx > 2: frame_idx+=1
    # if frame_idx > 4: frame_idx+=1
    # if frame_idx > 7: frame_idx+=1
    # if frame_idx > 19: frame_idx+=1
    # if frame_idx > 22: frame_idx+=1
    # if frame_idx > 24: frame_idx+=3

    org_curve, flt_curve = first_arrival_curve(canvas_out)
    # ax[0,0].plot(np.arange(canvas_out.shape[1]), org_curve, 'b', label='org out', alpha=0.2)
    # ax[0,0].plot(np.arange(canvas_out.shape[1]), flt_curve, 'b', label='output')
    out_flt_curve = flt_curve
    
    # Interpolate NaN values linearly
    nan_mask = np.isnan(out_flt_curve)
    x = np.arange(len(out_flt_curve))
    out_flt_curve[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], out_flt_curve[~nan_mask])
    return out_flt_curve

    # org_curve, flt_curve = first_arrival_curve(canvas_gt)
    # ax[0,0].plot(np.arange(canvas_out.shape[1]), org_curve, 'r', label='org gt', alpha=0.2)
    # ax[0,0].plot(np.arange(canvas_out.shape[1]), flt_curve, 'r', label='gt')

    
    # ax[0,0].plot(np.arange(canvas_out.shape[1]), out_flt_curve, 'y', label='interp')

    # ax[0,0].set_aspect('auto')
    # ax[0,0].set_title(f'Output (Frame {frame_idx+1}/40)')
    # ax[0,0].legend()
    
def get_fit_curves():
    # Create figure and axes once
    fit_curves = {}
    to_flip = False
    # fig, ax = plt.subplots(1, 1, figsize=(12,5), squeeze=False, sharex=True)
    data = lambda d: np.flip(d, axis=1) if to_flip else d


    # Prepare all frames first
    for i in np.arange(40)+1:
        n = stn_num_to_n[f"{i:02d}"]
        if n<0: continue
        file_index = ""
        canvas_inp = np.load(f'results/demultiple0120-waveform/canvas{file_index}_inp-{n}.npy')
        canvas_gt = np.load(f'results/demultiple0120-waveform/canvas{file_index}_gt-{n}.npy')
        canvas_out = np.load(f'results/demultiple0120-waveform/canvas{file_index}_model-final-{n}.npy')

        # Store data for each frame
        fit_curves[f'{(i):02d}'] = get_frame(data(canvas_out),data(canvas_gt))

    # # Create animation
    # anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1500, blit=False)
    # anim.save('animation.gif', writer='pillow')
    return fit_curves

fit_curves = get_fit_curves()