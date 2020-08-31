import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
sns.set()


# Plot Inputs 

Model_names = np.asarray([["Resnet20","Resnet32","Resnet44","Resnet110","DensenetBC-12","ResNeXt-50","EfficientNet-B0","EfficientNet-B2","EfficientNet-B4"],
               ["Resnet32","Resnet44","Resnet56","Resnet110","DensenetBC-12","ResNeXt-50","EfficientNet-B0","EfficientNet-B2","EfficientNet-B4"],
               ["Resnet20","Resnet32","Resnet44","Resnet110","DensenetBC-12","ResNeXt-50","EfficientNet-B0","EfficientNet-B2","EfficientNet-B4"],
               ["Resnet18","Resnet34","Resnet50","Resnet101","Densenet-121","ResNeXt-50"]
               ])

ensemble = np.asarray([[560,689,1023,2317,2718,2410,6120,7793,10143],
                       [579,712,1130,2581,2893,2502,6239,7982,10231],
                       [57,110,158,210,245,290,488,550,690],
                       [5320,5791,6129,6422,6682,7983]
                      ])

individual_sequential = np.asarray([[1223,2013,2485,6268,9231,9432,11582,13271,17342],
                                    [1321,2219,2561,6435,9451,9672,11712,13429,17564],
                                    [458,543,632,708,754,837,870,947,964],
                                    [7832,8672,9238,9564,10134,11245]
                                   ])

individual_parallel = np.asarray([[898,1232,1798,4578,6532,6823,8876,10298,13321],
                                    [976,1423,1987,4765,6956,7123,8978,10543,13564],
                                    [181,284,392,503,538,645,666,741,795],
                                    [6435,6987,7546,7934,8564,9456]
                                   ])

figure,axes = plt.subplots(2,2,figsize=(17,12))

# First Plot
line1, = axes[0,0].plot(ensemble[0],marker="o",label="Our Ensemble")
line2, = axes[0,0].plot(individual_sequential[0],marker="o",label="Individual students-Sequential")
line3, = axes[0,0].plot(individual_parallel[0],marker="o",label="Individual students-Parallel")
axes[0,0].legend()
axes[0,0].set_title("Cifar10")
axes[0,0].set_xticks(np.arange(9))
axes[0,0].set_xticklabels(Model_names[0],rotation=16)
axes[0,0].set_ylabel("Training time (mins)")

# Second Plot
line1, = axes[0,1].plot(ensemble[1],marker="o",label="Our Ensemble")
line2, = axes[0,1].plot(individual_sequential[1],marker="o",label="Individual students-Sequential")
line3, = axes[0,1].plot(individual_parallel[1],marker="o",label="Individual students-Parallel")
axes[0,1].legend()
axes[0,1].set_title("Cifar100")
axes[0,1].set_xticks(np.arange(9))
axes[0,1].set_xticklabels(Model_names[1],rotation=16)
axes[0,1].set_ylabel("Training time (mins)")

# Third Plot
line1, = axes[1,0].plot(ensemble[2],marker="o",label="Our Ensemble")
line2, = axes[1,0].plot(individual_sequential[2],marker="o",label="Individual students-Sequential")
line3, = axes[1,0].plot(individual_parallel[2],marker="o",label="Individual students-Parallel")
axes[1,0].legend()
axes[1,0].set_title("SVHN")
axes[1,0].set_xticks(np.arange(9))
axes[1,0].set_xticklabels(Model_names[2],rotation=16)
axes[1,0].set_ylabel("Training time (mins)")

# Fourth Plot
line1, = axes[1,1].plot(ensemble[3],marker="o",label="Our Ensemble")
line2, = axes[1,1].plot(individual_sequential[3],marker="o",label="Individual students-Sequential")
line3, = axes[1,1].plot(individual_parallel[3],marker="o",label="Individual students-Parallel")
axes[1,1].legend()
axes[1,1].set_title("Imagenet")
axes[1,1].set_xticks(np.arange(6))
axes[1,1].set_xticklabels(Model_names[3],rotation=16)
axes[1,1].set_ylabel("Training time (mins)")

plt.savefig("Training_time_comparison.pdf")