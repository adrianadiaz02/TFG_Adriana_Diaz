import torch

last_model_checkpoint_path = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs/bf_debug_tea_best/models/test._tea_!bg_dim40_ep4000_nm_lr0.001_mlp_size0_.pth.tar'
best_model_checkpoint_path = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs/bf_debug_tea/models/test._tea_!bg_dim40_ep4000_nm_lr0.001_mlp_size0_.pth.tar'

last_model_checkpoint = torch.load(last_model_checkpoint_path)
best_model_checkpoint = torch.load(best_model_checkpoint_path)

# Extract the epoch information
epoch_last = last_model_checkpoint.get('epoch')
epoch_best = best_model_checkpoint.get('epoch')

print(f"The baselline model was saved after epoch: {epoch_last}")
print(f"The best model was saved after epoch: {epoch_best}")


# Load the state dictionaries
last_model_state_dict = last_model_checkpoint.get('state_dict')
best_model_state_dict = best_model_checkpoint.get('state_dict')


# Initialize a flag to track if the models are different
models_differ = False

# Iterate through each parameter in the last model's state dictionary
for param_name in last_model_state_dict:
    # Compare the parameter with the best model's corresponding parameter
    if torch.equal(last_model_state_dict[param_name], best_model_state_dict[param_name]) == False:
        print(f"Difference found in parameter: {param_name}")
        models_differ = True
        break  # No need to check further if a difference is found

if not models_differ:
    print("The models are identical.")
else:
    print("The models are different.")



