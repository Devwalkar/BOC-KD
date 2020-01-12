import numpy as np 
import argparse 
import os 

parser = argparse.ArgumentParser(description='Parser for training')
parser.add_argument('--run_id',required=True,help='Run id of training experiment')
args = parser.parse_args()

Overall_path = os.path.join("../Model_storage",str(args.run_id),"Accuracy_arrays/Validation")

Valid_accuracy_arrays_names = [x for x in os.listdir(Overall_path) if "Valid_Accuracies_for_Student" in x]

print("------------ Evaluating the best validation accuracies for each student model")

for i,array_name in enumerate(Valid_accuracy_arrays_names):

    Accuracy_array = np.load(os.path.join(Overall_path,array_name),encoding="bytes")

    print("\nFor Student {} : Best validation accuracy : {:.3f} for [{} th /{}] observations".format(i+1,
                                                                                           np.max(Accuracy_array),
                                                                                           np.argmax(Accuracy_array)+1,
                                                                                           len(Accuracy_array)))