import numpy as np 
import argparse 
import os 

parser = argparse.ArgumentParser(description='Parser for training')
parser.add_argument('--run_id',required=True,help='Run id of training experiment')
parser.add_argument('--mode',default=1,help='Mode for running evaluator: 1:Framework comparision 2:Baseline comparisions')
parser.add_argument('--stu_run_id',default=None,help='Run id of baseline student experiments')
args = parser.parse_args()

Overall_path = os.path.join("../Model_storage",str(args.run_id),"Accuracy_arrays/Validation")

Valid_accuracy_arrays_names = sorted([x for x in os.listdir(Overall_path) if "Valid_Accuracies_for_Student" in x])

if int(args.mode) == 1:
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

elif int(args.mode) == 2:

    print("------------ Evaluating relative frmaework gain of student models over baseline")
    stu_run_id = args.stu_run_id.split("/")
    Student_array_names = sorted([os.path.join("../Model_storage",run_id,"Accuracy_arrays/Validation/Student_Valid_Accuracies.npy") for run_id in stu_run_id])

    for i,(valid_array,Stu_baseline) in enumerate(zip(Valid_accuracy_arrays_names,Student_array_names)):

        Framework_Student_max = np.max(np.load(os.path.join(Overall_path,valid_array),encoding="bytes"))
        Baseline_max = np.max(np.load(Stu_baseline,encoding="bytes"))

        print("\n For Student {} : Best validation accuracy : Baseline: {:.3f}% Framework: {:.3f}% Framework percent gain: {:.3f}%".format(i+1,
                                                                                                                               Baseline_max,
                                                                                                                               Framework_Student_max,
                                                                                                                               (float(Framework_Student_max-Baseline_max)*100)/Baseline_max)
                                                                                                                               )