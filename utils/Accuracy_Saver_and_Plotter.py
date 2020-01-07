import matplotlib.pyplot as plt 
import numpy as np 
import os 

def saver_and_plotter(Train_accuracies, 
            Val_accuracies,
			Train_losses,
			Train_ind_losses,
			Val_losses,
			Val_ind_losses,
			Store_root,
			run_id,
			No_students,
			plot_accuracy
			):

	# Saver function for training and validation accuracy/loss arrays and
	# Plotter function for plotting loss and accuracy curves for teacher and students

	# Train_accuracies shape : list of train accuracies for teacher, student 1 ....
	# Train_losses shape     : list of train losses for teacher, student 1 ....
	# Val_accuracies shape   : list of val accuracies for teacher, student 1 ....
	# Val_losses shape       : list of val losses for teacher, student 1 ....

	Loss_plot_names = ["Overall_Loss"]
	for i in range(No_students):
		Loss_plot_names.append("Student_{}".format(i+1))

	Accuracy_plot_names = ["Combined_Teacher"]
	for i in range(No_students):
		Accuracy_plot_names.append("Student_{}".format(i+1))


	for (train_accuracy,Val_accuracy,train_loss,Val_loss,loss_name,accuracy_name) in zip(Train_accuracies,
																						Val_accuracies,
																						Train_losses,
																						Val_losses,
																						Loss_plot_names,
																						Accuracy_plot_names):
		if plot_accuracy:

			train_accuracy = np.asarray(train_accuracy)
			Val_accuracy = np.asarray(Val_accuracy)

			train_loss = np.asarray(train_loss)
			Val_loss = np.asarray(Val_loss)

			assert len(train_loss) == len(Val_loss), "Loss arrays don't match !"
			Epochs = len(train_loss)
			A, = plt.plot(np.arange(1,Epochs+1,1),train_loss)
			B, = plt.plot(np.arange(1,Epochs+1,1),Val_loss)
			plt.xlabel("Number of Epochs")
			plt.ylabel("Epoch Loss")
			plt.legend(["Training set loss","Validation Set loss"])
			plt.title("Train and Val Loss over epochs for {}".format(loss_name))
			plt.savefig(os.path.join(Store_root,run_id,"Plots","Loss_comparision_{}.png".format(loss_name)))
			plt.clf()

			assert len(train_accuracy) == len(Val_accuracy), "Accuracy arrays don't match !"
			Epochs = len(train_accuracy)
			A, = plt.plot(np.arange(1,Epochs+1,1),train_accuracy)
			B, = plt.plot(np.arange(1,Epochs+1,1),Val_accuracy)
			plt.xlabel("Number of Epochs")
			plt.ylabel("Percent Accuracy")
			plt.legend(["Training set Accuracy","Validation Set Accuracy"])
			plt.title("Train and Val Accuracy over epochs for {}".format(accuracy_name))
			plt.savefig(os.path.join(Store_root,run_id,"Plots","Accuracy_comparision_{}.png".format(accuracy_name)))
			plt.clf()

    	# Saving accuracy and lose arrays 

		np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training',"Train_Accuracy_for_{}.npy".format(accuracy_name)),train_accuracy)
		np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation',"Valid_Accuracies_for_{}.npy".format(accuracy_name)),Val_accuracy)

		np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Training',"Train_loss_for_{}.npy".format(loss_name)),train_loss)
		np.save(os.path.join(Store_root,run_id,"Accuracy_arrays",'Validation',"Val_loss_for_{}.npy".format(loss_name)),Val_loss)


	# Saving Individual losses i.e. Normal loss, Representation loss and KL loss

	if plot_accuracy:

		Loss_names = ["Total_Normal_Loss","Total_Intermmediate_Loss","Total_KL_Loss"]

		for (train_loss,Val_loss,loss_name) in zip(Train_ind_losses,Val_ind_losses,Loss_names):

			train_loss = np.asarray(train_loss)
			Val_loss = np.asarray(Val_loss)

			assert len(train_loss) == len(Val_loss), "Loss arrays don't match !"
			Epochs = len(train_loss)
			A, = plt.plot(np.arange(1,Epochs+1,1),train_loss)
			B, = plt.plot(np.arange(1,Epochs+1,1),Val_loss)
			plt.xlabel("Number of Epochs")
			plt.ylabel("Epoch Loss")
			plt.legend(["Training set loss","Validation Set loss"])
			plt.title("Train and Val Loss over epochs for {}".format(loss_name))
			plt.savefig(os.path.join(Store_root,run_id,"Plots","Loss_comparision_{}.png".format(loss_name)))
			plt.clf()		