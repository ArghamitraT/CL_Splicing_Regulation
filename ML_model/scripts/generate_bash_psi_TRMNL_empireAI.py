"""
This files is used to submit files in the slurm
"""
import os
import time
import random

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")
def create_job_dir(dir="", fold_name = ""):
    if dir:
        job_path = os.path.join(dir, fold_name)
        os.mkdir(job_path)

    else:
        job_path = os.path.join(os.getcwd(), fold_name)
        if not os.path.exists(job_path):
            os.mkdir(job_path)

    return job_path


#job_path = create_job_dir(fold_name="job")

### it calls the .py file
def create_prg_file(prg_file_path):
   
    
    header = f"""#!/bin/bash
    set -e
    cd $HOME
    source ~/.bashrc
    conda activate cl_splicing_regulation3
    WORKDIR={data_dir}
    cd $WORKDIR
    python -m scripts.psi_regression_training \\
            task={task} \\
            task.val_check_interval={val_check_interval}\\
            task.global_batch_size={global_batch_size}\\
            trainer.max_epochs={max_epochs}\\
            tokenizer={tokenizer} \\
            tokenizer.seq_len={tokenizer_seq_len} \\
            embedder={embedder} \\
            loss={loss_name} \\
            embedder.maxpooling={maxpooling} \\
            optimizer={optimizer} \\
            optimizer.lr={learning_rate} \\
            aux_models.freeze_encoder={freeze_encoder} \\
            aux_models.train_mode={train_mode} \\
            aux_models.eval_weights={eval_weights} \\
            aux_models.warm_start={warm_start} \\
            aux_models.mtsplice_weights={mtsplice_weights} \\
            aux_models.mode={mtsplice_mode} \\
            aux_models.mtsplice_BCE={mtsplice_BCE} \\
            dataset.test_files.intronexon={test_file} \\
            ++wandb.dir="'{wandb_dir}'"\\
            ++logger.name="'{server_name}{slurm_file_name}{trimester}'"\\
            ++callbacks.model_checkpoint.dirpath="'{checkpoint_dir}'"\\
            ++hydra.run.dir={hydra_dir}\\
            ++logger.notes="{wandb_logger_NOTES}"
    """
    
    with open(prg_file_path, "w") as f:
        f.write(header)
    
    return prg_file_path
    
    
def copy_weights(desired_weight_path, to_be_saved_path ):
    for file_name in os.listdir(desired_weight_path):
        dir = os.path.join(desired_weight_path, file_name)
        os.system(f"cp {dir} {to_be_saved_path}")


def create_slurm_file(prg_file_path, job_name, slurm_file_path):

    show_name = '_'.join(job_name.split('_')[1:])
    show_name = f"{slurm_file_name}_{show_name}"

    header = f"#!/bin/bash\n" + \
    f"bash {prg_file_path}"

    with open (slurm_file_path, "w") as f:
        f.write(header)
    return slurm_file_path


def get_file_name(kind, l0=0, l1=0, l2=0, l3=0, ext=True):

    file_name = f"{kind}_{trimester}"
    if ext:
        file_name = f"{file_name}.sh"
    return file_name


server_name = 'EMPRAI'
server_path = '/mnt/home/at3836/'
main_data_dir = server_path+"Contrastive_Learning/files/results"
job_path = server_path+"Contrastive_Learning/files/cluster_job_submission_files"
code_dir = server_path+"Contrastive_Learning/code/ML_model"

data_dir_0   = create_job_dir(dir= main_data_dir, fold_name= "exprmnt"+trimester)
data_dir   = create_job_dir(dir= data_dir_0, fold_name= "files")
weight_dir = create_job_dir(dir= data_dir_0, fold_name="weights")
output_dir = create_job_dir(dir= data_dir, fold_name="output_files")
hydra_dir = create_job_dir(dir= data_dir, fold_name="hydra")
checkpoint_dir = create_job_dir(dir= weight_dir, fold_name="checkpoints")
wandb_dir = create_job_dir(dir= data_dir, fold_name="wandb")


""" Parameters: **CHANGE (AT)** """
slurm_file_name = 'psi_trial'
gpu_num = 1
hour = 1
memory = 100 # GB
nthred = 8 # number of CPU
task = "psi_regression_task" 
val_check_interval = 1.0
global_batch_size = 8196
embedder = "mtsplice"
tokenizer = "onehot_tokenizer"
loss_name = "MTSpliceBCELoss"
max_epochs = 2
maxpooling = True
optimizer = "sgd"
tokenizer_seq_len = 400
learning_rate =  1e-3
freeze_encoder = False  # Set to false for fine-tuning
warm_start = True
mtsplice_weights = "exprmnt_2025_07_30__13_10_26"
mtsplice_mode = "mtsplice"  
mtsplice_BCE = 1  # if instead of regression, you
train_mode = "eval"
eval_weights = "exprmnt_2025_08_17__02_17_03"  # Set to the directory of the weights you want to evaluate
            
# TRAIN_FILE="train_3primeIntron_filtered_min30views.pkl"
# VAL_FILE="val_3primeIntron_filtered.pkl"
# TEST_FILE="test_3primeIntron_filtered.pkl"

TEST_FILE="psi_test_Retina___Eye_psi_MERGED.pkl"

readme_comment = (
     "psi trial"
)
wandb_logger_NOTES="psi trial" ## do NOT use any special character or new line

""" Parameters: **CHANGE (AT)** """ 

# train_file = server_path+"Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"+TRAIN_FILE
# val_file = server_path+"Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"+VAL_FILE
test_file = server_path+"Contrastive_Learning/data/final_data/ASCOT_finetuning//"+TEST_FILE


name = slurm_file_name

def create_readme():
    name = os.path.join(data_dir, "readme")
    readme = open(name, "a")
    comment = readme_comment
    readme.write(comment)
    readme.close()



def gen_combination():
    
    create_readme()

    kind = name
    # python_file_path = os.path.join(code_dir, name)

    hash_obj = random.getrandbits(25)
    
    prg_file_path = os.path.join(job_path, get_file_name(kind= f"prg_{kind}_{hash_obj}"))
    slurm_file_path = os.path.join(job_path, get_file_name(kind= f"slurm_{kind}_{hash_obj}"))
    
    # create_prg_file(python_file_path=python_file_path, prg_file_path=prg_file_path, output_file_path=output_file_path, input_file_names=set, alpha_initial=alpha_val)
    create_prg_file(prg_file_path=prg_file_path) 

    job_name=get_file_name(kind=f"{kind}_{hash_obj}", ext=False)
    
    create_slurm_file(prg_file_path=prg_file_path, 
                    job_name=job_name, 
                    slurm_file_path=slurm_file_path)

    os.system(f"cp -r {code_dir}/scripts {data_dir}")
    os.system(f"cp -r {code_dir}/configs {data_dir}")
    os.system(f"cp -r {code_dir}/src {data_dir}")
    
    #os.system(f"cp {utility_file_path} {data_dir}")
    # (AT)
    os.system(f"chmod u+x {prg_file_path}")
    os.system(f"bash {prg_file_path} 2>&1 | tee {output_dir}/out_{job_name}.txt")
    # os.system(f"chmod u+x {slurm_file_path}")
    # os.system(f"sbatch {slurm_file_path}")
                    

def main():
    gen_combination()

if __name__ == "__main__":
    main()