

# Process:
In this folder we have the code to run the experiments described at the paper.


## Preprocess synthetic data and train without symmetries

We can train our first model on the synthetic data using the code at 
First, we need to preprocess the simulated data and save it into a local folder, using *preprocess_data_synth.py* with the variable *using_symmetries_inferred = False*. If we want to use the split with different objects, we can use the code at *change_split_epson_diffobjects.py*.

## Train to predict symmetries on synthetic and real CAD models

Having preprocessed the real and synthetic data, we can train the model to predict symmetries with the file 
*train_predict_symmetries.py*.

Then, we can predict symmetries for all our CAD models using *predict_symmetries_synth.py*.

## Preprocess again and train using symmetries

Now we can use the predicted symmetries to train a more powerful model. Similarly to the first step, we can modify the folder of our data and train the new model using *preprocess_data_synth.py* with *using_symmetries_inferred = True*.

## Train on Real data

Now that we have a model trained on several CAD models, we can finally use the accumulated knowledge into real data using the script *train_real_data.py*.

## Check results

The results can be obtained through the script *get_test_results.py*.

## Properties of the objects and poses

You can obtain a few details on the data and view properties using the following scripts. The code in *get_distribution_ground_truth.py* shows the distribution of the ground truth poses. The scripts *transforms.py* *utils_trainsymmetry.py* *rotm2quat.py* and *eval_pair_symm.py* summarize the theoretical properties that we have used for improving our pose estimation algorithms. 
