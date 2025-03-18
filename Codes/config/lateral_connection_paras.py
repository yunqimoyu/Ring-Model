import numpy as np

sigma_0 = 24
sigma_narrow = sigma_0/np.sqrt(2)
sigma_wide = sigma_0*np.sqrt(2)
enhance_peak = 2

def amplitude_get_by_peak(enhance_peak, sigma):
    return (1 - 1/enhance_peak)/(np.sqrt(2*np.pi)*sigma)

def get_kernel_para(amplitude, sigma=sigma_narrow):
    kernel_para = {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma, 'amplitude': amplitude}}
    return kernel_para

def kernel_para_get_by_peak(enhance_peak = enhance_peak, sigma = sigma_0):
    return get_kernel_para(amplitude_get_by_peak(enhance_peak, sigma), sigma)

def EI_amp_critical_value(sigma_EE=sigma_narrow,
                          sigma_IE=sigma_narrow, alpha_EE = amplitude_get_by_peak(enhance_peak, sigma_narrow)):
    denominator = np.sqrt(2*np.pi)*sigma_IE**2*sigma_IE**2*2
    alpha_square = (sigma_EE**3*alpha_EE)/denominator
    alpha = np.sqrt(alpha_square)
    return alpha

amp_EE = amplitude_get_by_peak(enhance_peak, sigma_0)
amp_EE_narrow = amp_EE
amp_EE_wide = amp_EE

EI_amp = np.sqrt(sigma_0*amp_EE/np.sqrt(2*np.pi))/sigma_narrow 
EI_amp_critical_EE_narrow = EI_amp_critical_value(sigma_EE=sigma_narrow, alpha_EE=amp_EE_narrow)
EI_amp_critical_EE_wide = EI_amp_critical_value(sigma_EE = sigma_wide, alpha_EE = amp_EE_wide)

kEE = get_kernel_para(amplitude=amp_EE, sigma=sigma_0)
kEE_narrow = get_kernel_para(amplitude=amp_EE_narrow, sigma=sigma_narrow)
kEE_wide = get_kernel_para(amplitude=amp_EE_wide, sigma=sigma_wide)


EN0_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}}}
EN1_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_narrow+EI_amp)/2}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_narrow+EI_amp)/2}}}
EN2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}}}


EN0_v2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow}},
               'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}},
               'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}}}
EN1_v2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow*4}},
               'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_narrow+EI_amp)}},
               'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_narrow+EI_amp)}}}
EN2_v2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': amp_EE_narrow*4}},
               'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp*2.3}},
               'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp*2.3}}}


SW0_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_0,      'amplitude': amp_EE}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_narrow/1.1}}}
SW1_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_0,      'amplitude': amp_EE}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}}}
SW2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_0,      'amplitude': amp_EE}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_wide*1.1}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_wide*1.1}}}


EW0_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_wide,   'amplitude': amp_EE_wide}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp}}}
EW1_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_wide,   'amplitude': amp_EE_wide}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_wide + EI_amp)/2}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': (EI_amp_critical_EE_wide + EI_amp)/2}}}
EW2_para = {'kEE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_wide,   'amplitude': amp_EE_wide}},
            'kEI': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_wide*1.1}},
            'kIE': {'type': 'gaussian', 'para': {'mu': 0, 'sigma': sigma_narrow, 'amplitude': EI_amp_critical_EE_wide*1.1}}}

params_6 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 5, 'amplitude': 0.004}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 5, 'amplitude': 0.02}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 5, 'amplitude': 0.02}}}

params_7 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 24, 'amplitude': 4}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 24, 'amplitude': 2}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 24, 'amplitude': 2}}}

params_8 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 8, 'amplitude': 0.01*128/180}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 8, 'amplitude': 0.01*128/180}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 8, 'amplitude': 0.01*128/180}}}


params_9 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.025}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.001}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.001}}}
            
params_10 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.08}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.05}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.05}}}

params_11 = {     
            'kEE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.16}},
            'kIE': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.1}},
            'kEI': {'type': 'gaussian', 'para':{'mu': 0, 'sigma': 10, 'amplitude': 0.1}}}

lateral_cnct_params_dict = {'EN0': EN0_para, 'EN1': EN1_para, 'EN2': EN2_para, 
                            'EN0_v2': EN0_v2_para, 'EN1_v2': EN1_v2_para, 'EN2_v2': EN2_v2_para,
                            'SW0': SW0_para, 'SW1': SW1_para, 'SW2': SW2_para,
                            'EW0': EW0_para, 'EW1': EW1_para, 'EW2': EW2_para, 
                            '06':params_6, '07':params_7, '08':params_8, '09':params_9, '10':params_10, '11':params_11}