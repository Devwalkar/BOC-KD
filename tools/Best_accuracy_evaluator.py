import numpy as np 
import argparse 
import os 

parser = argparse.ArgumentParser(description='Parser for training')
parser.add_argument('--run_id',required=True,help='Run id of training experiment')
args = parser.parse_args()

Overall_path = os.path.join("../Model_storage",str(args.run_id),"Accuracy_arrays/Validation")

Valid_accuracy_arrays_names = sorted([x for x in os.listdir(Overall_path) if "Valid_Accuracies_for_Student" in x])

print("------------ Evaluating the best validation accuracies for each student model")

Combined_Teacher_array = np.load(os.path.join(Overall_path,"Valid_Accuracies_for_Combined_Teacher.npy"),encoding="bytes")
Combined_Teacher_max = np.max(Combined_Teacher_array)
print("\nCombined Teacher Best Validation Accuracy: {:.3f} %".format(Combined_Teacher_max))

pseduo_teacher_max = np.max(np.load(os.path.join(Overall_path,Valid_accuracy_arrays_names[0]),encoding="bytes"))

for i,array_name in enumerate(Valid_accuracy_arrays_names):

    Accuracy_array = np.load(os.path.join(Overall_path,array_name),encoding="bytes")

    print("\nFor Student {} : Best validation accuracy : {:.3f} % for [{} th /{}] observations"
    "    Pseudo Teacher Gain: {:.3f} %     Teacher Gain: {:.3f} %".format(i+1,
                                                                        np.max(Accuracy_array),
                                                                        np.argmax(Accuracy_array)+1,
                                                                        len(Accuracy_array),
                                                                        (float(np.max(Accuracy_array) -pseduo_teacher_max)*100)/pseduo_teacher_max,
                                                                        (float(np.max(Accuracy_array) -Combined_Teacher_max)*100)/Combined_Teacher_max)
                                                                        )