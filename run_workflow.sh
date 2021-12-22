#
# Sample workflow shell script to run training and eval scripts
#
# change train and eval flags to run one w/out the other
train=1
eval=1
# workflow path is where these files all sit, likely this directory
workflow_path='.'
#name of the training script, change acordingly
train_script="Train_3d.py"
# name of the eval script, change accordingly
eval_script='Eval_3D.py'
# the yaml file that sets the training and eval datasets and ML model def
yaml='workflow.yaml'
# a model / run name for this ML model
config_name='model-TV-3D'

if [ $train == 1 ]
then
    echo 'running training'
    python3 $workflow_path$train_script $yaml $config_name
fi

if [ $eval == 1 ]
then
    echo 'running evaluation'
    python3 $workflow_path$eval_script $yaml $config_name
fi

