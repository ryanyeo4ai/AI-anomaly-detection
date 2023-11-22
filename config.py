#mode_names = ['LAM', 'SP', 'TP', 'TSH', 'WOL']
#mode_names = ['Laundry']
mode_names = ['PM25']
col_names = 16 #PM25
#mode_names = ['TP']
scenario_names = ['1']
#scenario_names = ['MT2', 'MT3', 'MT4']
#scenario_names = ['MT1', 'TSH1', 'TSH2']
#scenario_names = ['LAM1']


# data
data_selector = 'mode'  # mode or scenario
train_base_path = './train_normal'
test_base_path = './test_normal'
eval_base_path = './test_abnormal/'
#eval_base_path_ryan = './test_abnormal/normalization_sample/'
train_data_list = './train_list/'

train_data = './data_csv/'
test_data = './test_normal/'

split_excel_name = 'train_data_list.xlsx'

#auxiliary_path_for_stats = ['08_RWA_speed-D']  # use some data in these anomaly data in computing min/max stats for normalization
auxiliary_path_for_stats = ['normalization_sample']  # use some data in these anomaly data in computing min/max stats for normalization
warm_up_period = 1  # exclude first N timesteps
n_scenario = 15 # number of scenario

sample_length = 8
sample_stride = 4

# see 'calc_mean_var.py' load_csv function
exclude_columns = []  # no exclusion
#exclude_columns = ['acs41_a003', 'acs32_a023', 'acs71_a043', 'acs11_a006', 'acs23_a007', 'acs31_a000', 'acs33_a000', 'acs41_a000', 'perf_a043']  # discrete columns

# training
epochs = 10
batch_size = 256
log_interval = 5
lr = 1e-3 #학습률(VAE)
bayes_vae_lr = 1e-3 # 베이지안 최적화 결과 넣는 학습률(VAE)
#lamda = 0.5 #람다값(KL_div, VAE)
bayes_vae_lamda = 0.5 # 베이지안 최적화 결과 넣는 람다값(KL_div, VAE)
bayes_vae_warm_up_period = 200
bayes_ganno_lr = 1e-3 # 베이지안 최적화 결과 넣는 학습률(GANomaly)
bayesian = True # if True, applying bayesian optimization(train_VAE.py)

# output
out_path = './results'

def calc_dim(dim, param):
    if (int(dim / param) > 0):
        return int(dim / param)
    else:
        return 1