import json
import yaml
import itertools
import subprocess
from copy import deepcopy
import subprocess
import signal
import os

from tqdm import tqdm

def load_config(file_path):
    """
    Load configuration from a JSON or YAML file.
    Args:
        file_path: Path to the configuration file.
    Returns:
        Config dictionary.
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .yaml.")


def extend_command(config, command):

    """
    Constructs the command line arguments for evaluation.py from the config dictionary.

    Args:
        config: A dictionary containing all parameters from the YAML file.

    Returns:
        A list of strings representing the command line arguments.
    """

    for key, value in config.items():
        if isinstance(value, (list, tuple)):
      # Handle list/tuple arguments by converting them to space-separated strings
            value_str = " ".join(str(item) for item in value)
        elif isinstance(value, bool):
            if value:
                value_str = ""
            else:
                continue
        else:
            value_str = str(value)
        command.extend([f"--{key}", value_str])
    return command


if __name__ == "__main__":
    # Load configuration
    config_file = "fuse_then_eval_config_24.yaml"  
    log_file = "/data/rech/huiyuche/TREC_iKAT_2024/logs/evaluation_log_2023.txt"
    

    config = load_config(config_file)

    # Extract parameters and experiment config
    fixed_parameters_dict = config["fixed"]
    to_iterate_parameters_dict = config["iterate"]

    initial_command = ["python", "evaluation.py"]

    # iterate over all parameter combinations (use itertools.product)
    # Generate all parameter combinations (Cartesian product)
    param_keys = to_iterate_parameters_dict.keys()
    param_values = [to_iterate_parameters_dict[key] for key in param_keys]
    param_combinations = itertools.product(*param_values)


    processes = []
    try:
        for combination in param_combinations:
            print("begin")
            param_dict = dict(zip(param_keys, combination))
            fixed_parameters_dict_copy = deepcopy(fixed_parameters_dict)

            if param_dict["topics"] == "ikat_24_test":
                fixed_parameters_dict_copy["input_query_path"] = fixed_parameters_dict_copy["input_query_path"].replace("23", "24")
                fixed_parameters_dict_copy["qrel_file_path"] = fixed_parameters_dict_copy["qrel_file_path"].replace("23", "24")
            if param_dict["topics"] == "ikat_23_test":
                fixed_parameters_dict_copy["input_query_path"] = fixed_parameters_dict_copy["input_query_path"].replace("24", "23")
                fixed_parameters_dict_copy["qrel_file_path"] = fixed_parameters_dict_copy["qrel_file_path"].replace("24", "23")

            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_judge_and_rewrite_rw":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_judge_and_rewrite_rw"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.4]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_judge_and_rewrite_rwrs":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_judge_and_rewrite_rwrs"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.1]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_rar_rwrs":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_rar_rwrs"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.1]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rw":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rwrs", "gpt-4o_judge_and_rewrite_rw"]
                fixed_parameters_dict_copy["fuse_weights"] = [0.1,0.4]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rwrs":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rwrs", "gpt-4o_judge_and_rewrite_rwrs"]
                fixed_parameters_dict_copy["fuse_weights"] = [0.1,0.1]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_judge_and_rewrite_rw":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_rar_rwrs", "gpt-4o_judge_and_rewrite_rw"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.1,0.4]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_judge_and_rewrite_rwrs":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_rar_rwrs", "gpt-4o_judge_and_rewrite_rwrs"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.1,0.1]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rw_fuse_judge_and_rewrite_rwrs_fuse_judge_and_rewrite_rw":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rw", "gpt-4o_judge_and_rewrite_rwrs", "gpt-4o_judge_and_rewrite_rw"]
                fixed_parameters_dict_copy["fuse_weights"] = [1,0.1,0.4]
            if param_dict["retrieval_query_type"] == "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rwrs_fuse_judge_and_rewrite_rw":
                fixed_parameters_dict_copy["QRs_to_rank"] = ["gpt-4o_rar_rwrs", "gpt-4o_judge_and_rewrite_rwrs", "gpt-4o_judge_and_rewrite_rw"]
                fixed_parameters_dict_copy["fuse_weights"] = [0.1,0.1,0.4]
            

                
            base_command = extend_command(fixed_parameters_dict_copy, initial_command)
            # distinguish ikat 23 and 24
            command = extend_command(param_dict, base_command)
            command.append(f"&>> {log_file}")
            print("running experiment with parameters: ", param_dict)
            command = " ".join(command)
            
            # Start child processes and record
            process = subprocess.Popen(command, shell=True)
            processes.append(process)
            process.wait()
        
    except KeyboardInterrupt:
        print("Ctrl+C detected, terminating child processes...")
        for process in processes:
            if process and process.poll() is None:  # Check if process is still running
                os.kill(process.pid, signal.SIGKILL)  # Forcefully kill the process
                process.wait()  # Wait for it to terminate
            
            



