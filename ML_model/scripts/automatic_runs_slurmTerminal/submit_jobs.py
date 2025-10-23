"""
This files is used to submit files in the slurm
"""
import os
import time
import random


############# DEBUG Message ###############
import inspect
import os
_warned_debug = False  # module-level flag
def reset_debug_warning():
    global _warned_debug
    _warned_debug = False
def debug_warning(message):
    global _warned_debug
    if not _warned_debug:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        print(f"\033[1;31m⚠️⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️  DEBUG MODE ENABLED in {filename}:{lineno} —{message} REMEMBER TO REVERT!\033[0m")
        _warned_debug = True
############# DEBUG Message ###############


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

def create_bash_file(header, bash_file_path):

    with open(bash_file_path, "w") as f:
        f.write(header)

    return bash_file_path

def copy_weights(desired_weight_path, to_be_saved_path ):
    for file_name in os.listdir(desired_weight_path):
        dir = os.path.join(desired_weight_path, file_name)
        os.system(f"cp {dir} {to_be_saved_path}")

def get_file_name(kind, l0=0, l1=0, l2=0, l3=0, ext=True):

    file_name = f"{kind}_{trimester}"
    if ext:
        file_name = f"{file_name}.sh"
    return file_name

import os

def get_server_paths():

    cwd = os.path.abspath(os.getcwd())
    target = "Contrastive_Learning"

    # Search upward until we find 'Contrastive_Learning' folder
    parts = cwd.split(os.sep)
    base_path = None
    for i in range(len(parts), 0, -1):
        candidate = os.sep.join(parts[:i])
        if os.path.basename(candidate) == target:
            base_path = candidate
            break
        elif os.path.isdir(os.path.join(candidate, target)):
            base_path = os.path.join(candidate, target)
            break

    if base_path is None:
        raise FileNotFoundError("Could not locate 'Contrastive_Learning' base directory.")

    # Identify the server path (folder *before* Contrastive_Learning)
    server_path = os.path.dirname(base_path) + os.sep
    folder_before = os.path.basename(os.path.normpath(server_path))

    if folder_before == "atalukder":
        server_name = "NYGC"
    elif folder_before == "at3836":
        server_name = "EMPRAI"

    
    # Build standard subpaths
    main_data_dir = os.path.join(base_path, "files", "results")
    job_path = os.path.join(base_path, "files", "cluster_job_submission_files")
    code_dir = os.path.join(base_path, "code", "ML_model")

    return {
        "server_name": server_name,
        "server_path": server_path,
        "main_data_dir": main_data_dir,
        "job_path": job_path,
        "code_dir": code_dir,
    }


def create_readme(data_dir, readme_comment):
    p = os.path.join(data_dir, "readme")
    with open(p, "a") as f:
        f.write(readme_comment)


#############

def create_prg_header_cl(cfg, paths):

    train_file = paths["server_path"]+"Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"+cfg["TRAIN_FILE"]
    val_file = paths["server_path"]+"Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"+cfg["VAL_FILE"]
    test_file = paths["server_path"]+"Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"+cfg["TEST_FILE"]

    
    header = f"""#!/bin/bash
    set -e
    cd $HOME
    source ~/.bashrc
    conda activate cl_splicing_regulation3
    WORKDIR={paths["data_dir"]}
    cd $WORKDIR
    python -m scripts.cl_training \\
            task={cfg["task"]} \\
            task.val_check_interval={cfg["val_check_interval"]}\\
            task.global_batch_size={cfg["global_batch_size"]}\\
            trainer.max_epochs={cfg["max_epochs"]}\\
            tokenizer={cfg["tokenizer"]} \\
            embedder={cfg["embedder"]} \\
            loss={cfg["loss_name"]} \\
            optimizer={cfg["optimizer"]} \\
            dataset.n_augmentations={cfg["n_augmentations"]} \\
            embedder.maxpooling={cfg["maxpooling"]} \\
            embedder.seq_len={cfg["tokenizer_seq_len"]}\\
            dataset.fixed_species={cfg["fixed_species"]} \\
            dataset.train_data_file={train_file} \\
            dataset.val_data_file={val_file} \\
            dataset.test_data_file={test_file} \\
            ++wandb.dir="'{paths["wandb_dir"]}'"\\
            ++logger.name="'{paths["server_name"]}{cfg["slurm_file_name"]}{trimester}'"\\
            ++callbacks.model_checkpoint.dirpath="'{paths["checkpoint_dir"]}'"\\
            ++hydra.run.dir={paths["hydra_dir"]}\\
            ++logger.notes="{cfg["wandb_logger_NOTES"]}"
    """
    # reset_debug_warning()
    # debug_warning("add those args later")
   
    # tokenizer.seq_len={tokenizer_seq_len} \\
    return header


def create_slurm_header_cl(cfg, paths):
    
    #### need NYGC vs EMIPIRE AI condition here later #### (AT)
    show_name = '_'.join(cfg["job_name"].split('_')[1:])
    show_name = f"{cfg['slurm_file_name']}_{show_name}"

    if paths['server_name'] == "EMPRAI":
        header = f"#!/bin/bash\n" + \
        "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n" + \
        "##NECESSARY JOB SPECIFICATIONS\n" + \
        f"#SBATCH --job-name={show_name}      #Set the job name to \"JobExample1\"\n" + \
        "#SBATCH --partition=columbia     \n" + \
        "#SBATCH --account=columbia     \n" + \
        f"#SBATCH --gres=gpu:{cfg['gpu_num']}     \n" + \
        f"#SBATCH --time={cfg['hour']}:45:00              #Set the wall clock limit \n" + \
        f"#SBATCH --mem={cfg['memory']}G              \n" + \
        f"#SBATCH --cpus-per-task={cfg['nthred']}                   \n" + \
        "#SBATCH --mail-type=END,FAIL    \n" + \
        f"#SBATCH --output={paths['output_dir']}/out_{cfg['job_name']}.%j      #Send stdout/err to\n" + \
        "#SBATCH --mail-user=at3836@columbia.edu                    \n" + \
         f"{paths['prg_file_path']}"
    
    elif paths['server_name'] == "NYGC":
        header = f"#!/bin/bash\n" + \
        "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n" + \
        "##NECESSARY JOB SPECIFICATIONS\n" + \
        f"#SBATCH --job-name={show_name}      #Set the job name to \"JobExample1\"\n" + \
        "#SBATCH --partition=gpu     \n" + \
        f"#SBATCH --gres=gpu:{cfg['gpu_num']}     \n" + \
        f"#SBATCH --time={cfg['hour']}:45:00              #Set the wall clock limit \n" + \
        f"#SBATCH --mem={cfg['memory']}G              \n" + \
        f"#SBATCH --cpus-per-task={cfg['nthred']}                   \n" + \
        "#SBATCH --mail-type=END,FAIL    \n" + \
        f"#SBATCH --output={paths['output_dir']}/out_{cfg['job_name']}.%j      #Send stdout/err to\n" + \
        "#SBATCH --mail-user=atalukder@nygenome.org                    \n" + \
        f"{paths['prg_file_path']}"


    return header



def create_prg_header_psi(cfg, paths):
    # NOTE: we will export RUN_IDX in the Slurm script loop.
    # We suffix logger.name (and optionally hydra.run.dir) with _r$RUN_IDX
    test_file = paths["server_path"]+"Contrastive_Learning/data/final_data/ASCOT_finetuning/"+cfg["PSI_TEST_FILE"]
    
    header = f"""#!/bin/bash
    set -e
    cd $HOME
    source ~/.bashrc
    conda activate cl_splicing_regulation3
    : "${{RUN_IDX:=1}}"
    : "${{NEW_WANDB_PROJECT:=0}}"    # 1 => one new W&B project for ALL runs in this job
    PROJECT_NAME="{paths["server_name"]}{cfg["slurm_file_name"]}{trimester}"   # no _run_ suffix
    EXTRA_WANDB_PROJECT=""
    if [ "$NEW_WANDB_PROJECT" = "1" ]; then
      EXTRA_WANDB_PROJECT=++logger.project="$PROJECT_NAME"
    fi
    WORKDIR={paths["data_dir"]}
    cd $WORKDIR
    python -m scripts.psi_regression_training \\
            task={cfg["task"]} \\
            task.val_check_interval={cfg["val_check_interval"]}\\
            task.global_batch_size={cfg["global_batch_size"]}\\
            trainer.max_epochs={cfg["max_epochs"]}\\
            tokenizer={cfg["tokenizer"]} \\
            embedder={cfg["embedder"]} \\
            loss={cfg["psi_loss_name"]} \\
            optimizer={cfg["optimizer"]} \\
            optimizer.lr={cfg["learning_rate"]} \\
            aux_models.freeze_encoder={cfg["freeze_encoder"]} \\
            aux_models.warm_start={cfg["warm_start"]} \\
            aux_models.mtsplice_weights={cfg["mtsplice_weights"]} \\
            aux_models.weights_threeprime={cfg["weight_3p"]} \\
            aux_models.weights_fiveprime={cfg["weight_5p"]} \\
            aux_models.weights_exon={cfg["weight_exon"]} \\
            embedder.maxpooling={cfg["maxpooling"]} \\
            tokenizer.seq_len={cfg["tokenizer_seq_len"]} \\
            aux_models.mode={cfg["mode"]} \\
            aux_models.mtsplice_BCE={cfg["mtsplice_BCE"]} \\
            dataset.test_files.intronexon={test_file} \\
            ++wandb.dir="'{paths["wandb_dir"]}'"\\
            $EXTRA_WANDB_PROJECT \\
            ++logger.name="'{paths["server_name"]}{cfg["slurm_file_name"]}{trimester}_run_$RUN_IDX'"\\
            ++callbacks.model_checkpoint.dirpath="'{paths["checkpoint_dir"]}/run_$RUN_IDX'"\\
            ++hydra.run.dir="{paths["hydra_dir"]}/run_$RUN_IDX"\\
            ++logger.notes="{cfg["wandb_logger_NOTES"]}"
    """

    # reset_debug_warning()
    # debug_warning("add those args later")
    #

    return header

def create_slurm_header_psi(cfg, paths):
    
    #### need NYGC vs EMIPIRE AI condition here later #### (AT)
    show_name = "_".join(cfg["job_name"].split("_")[1:])
    show_name = f"{cfg['slurm_file_name']}_{show_name}"

    if paths['server_name'] == "EMPRAI":
        header = (
        "#!/bin/bash\n"
        "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n"
        "##NECESSARY JOB SPECIFICATIONS\n"
        f"#SBATCH --job-name={show_name}\n"
        "#SBATCH --partition=columbia\n"
        "#SBATCH --account=columbia\n"
        f"#SBATCH --gres=gpu:{cfg['gpu_num']}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task={cfg['nthred']}\n"
        f"#SBATCH --time={cfg['hour']}:30:00\n"
        f"#SBATCH --mem={cfg['memory']}G\n"
        "#SBATCH --mail-type=END,FAIL\n"
        f"#SBATCH --output={paths['output_dir']}/out_{cfg['job_name']}.%j      # job-wide stdout/err\n"
        "#SBATCH --mail-user=at3836@columbia.edu\n"
        "\n"
        f"RUNS={cfg['run_num']}\n"
        f"NEW_WANDB_PROJECT={cfg['new_project_wandb']}\n"   # set once; all runs share one project if =1
        "echo \"SLURM_JOB_ID=$SLURM_JOB_ID\"\n"
        "for i in $(seq 1 $RUNS); do\n"
        "  export RUN_IDX=$i\n"
        "  export NEW_WANDB_PROJECT=$NEW_WANDB_PROJECT\n"
        "  echo \"===== [$(date)] Starting RUN $i/$RUNS on job $SLURM_JOB_ID =====\"\n"
        f"  bash {paths['prg_file_path']} 2>&1 | tee {paths['output_dir']}/out_{cfg['job_name']}_r${{i}}.${{SLURM_JOB_ID}}.txt\n"
        "  status=${PIPESTATUS[0]}\n"
        "  echo \"===== [$(date)] Finished RUN $i/$RUNS with status ${status} =====\"\n"
        "  if [[ $status -ne 0 ]]; then\n"
        "    echo \"Run $i failed (status $status). Exiting early.\" >&2\n"
        "    exit $status\n"
        "  fi\n"
        "done\n"
    )

    elif paths['server_name'] == "NYGC":
        header = (
            "#!/bin/bash\n"
            "## SLURM settings\n"
            f"#SBATCH --job-name={show_name}\n"
            "#SBATCH --partition=gpu\n"
            f"#SBATCH --gres=gpu:{cfg['gpu_num']}\n"
            f"#SBATCH --time={cfg['hour']}:45:00\n"
            f"#SBATCH --mem={cfg['memory']}G\n"
            f"#SBATCH --cpus-per-task={cfg['nthred']}\n"
            "#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --output={paths['output_dir']}/out_{cfg['job_name']}.%j\n"
            "#SBATCH --mail-user=atalukder@nygenome.org\n"
            "\n"
            f"RUNS={cfg['run_num']}\n"
            f"NEW_WANDB_PROJECT={cfg['new_project_wandb']}\n"   # set once; all runs share one project if =1
            "echo \"SLURM_JOB_ID=$SLURM_JOB_ID\"\n"
            "for i in $(seq 1 $RUNS); do\n"
            "  export RUN_IDX=$i\n"
            "  export NEW_WANDB_PROJECT=$NEW_WANDB_PROJECT\n"
            "  echo \"===== [$(date)] Starting RUN $i/$RUNS on job $SLURM_JOB_ID =====\"\n"
            f"  bash {paths['prg_file_path']} 2>&1 | tee {paths['output_dir']}/out_{cfg['job_name']}_r${{i}}.${{SLURM_JOB_ID}}.txt\n"
            "  status=${PIPESTATUS[0]}\n"
            "  echo \"===== [$(date)] Finished RUN $i/$RUNS with status ${status} =====\"\n"
            "  if [[ $status -ne 0 ]]; then\n"
            "    echo \"Run $i failed (status $status). Exiting early.\" >&2\n"
            "    exit $status\n"
            "  fi\n"
            "done\n"
        )
    return header




# --------------------------------------------------------------------
# 2. PATH SETUP (automatically detects server)
# --------------------------------------------------------------------
def setup_paths():
    """Automatically detect server and create directory structure."""
    paths = get_server_paths()
    server_name = paths["server_name"]
    server_path = paths["server_path"]

    main_data_dir = paths["main_data_dir"]
    job_path = paths["job_path"]
    code_dir = paths["code_dir"]

    data_dir_0 = create_job_dir(dir=main_data_dir, fold_name="exprmnt" + trimester)
    data_dir = create_job_dir(dir=data_dir_0, fold_name="files")
    weight_dir = create_job_dir(dir=data_dir_0, fold_name="weights")
    output_dir = create_job_dir(dir=data_dir, fold_name="output_files")
    hydra_dir = create_job_dir(dir=data_dir, fold_name="hydra")
    checkpoint_dir = create_job_dir(dir=weight_dir, fold_name="checkpoints")
    wandb_dir = create_job_dir(dir=data_dir, fold_name="wandb")

    return dict(
        server_name=server_name,
        server_path=server_path,
        main_data_dir=main_data_dir,
        job_path=job_path,
        code_dir=code_dir,
        data_dir=data_dir,
        weight_dir=weight_dir,
        output_dir=output_dir,
        hydra_dir=hydra_dir,
        checkpoint_dir=checkpoint_dir,
        wandb_dir=wandb_dir,
    )



def submit_job(cfg, paths):
    """Create program + slurm script and submit."""
    
    kind = cfg["slurm_file_name"]
    hash_obj = random.getrandbits(25)

    prg_file = os.path.join(paths["job_path"], get_file_name(f"prg_{kind}_{hash_obj}"))
    slurm_file = os.path.join(paths["job_path"], get_file_name(f"slurm_{kind}_{hash_obj}"))

    paths['prg_file_path'] = prg_file
    paths['slurm_file_path'] = slurm_file
   
    os.system(f"cp -r {paths['code_dir']}/scripts {paths['data_dir']}")
    os.system(f"cp -r {paths['code_dir']}/configs {paths['data_dir']}")
    os.system(f"cp -r {paths['code_dir']}/src {paths['data_dir']}")
    

    job_name = get_file_name(f"{kind}_{hash_obj}", ext=False)
    cfg["job_name"] = job_name

    task = cfg["task"] # "psi_regression_task" or "introns_cl"
    if task == "introns_cl":
        prg_header =  create_prg_header_cl(cfg, paths)
        slurm_header =  create_slurm_header_cl(cfg, paths)
    elif task == "psi_regression_task":
        prg_header =  create_prg_header_psi(cfg, paths)
        slurm_header =  create_slurm_header_psi(cfg, paths)

    create_bash_file(prg_header, prg_file)
    create_bash_file(slurm_header, slurm_file)

    os.system(f"chmod u+x {prg_file}")
    os.system(f"chmod u+x {slurm_file}")
    os.system(f"sbatch {slurm_file}")



