import os 


Train_Files = sorted(os.listdir("data/Caltech/Train/256_ObjectCategories"))
Test_Files = sorted(os.listdir("data/Caltech/Test/256_ObjectCategories"))

for i,file in enumerate(Train_Files):

    Images = os.listdir(os.path.join("data/Caltech/Train/256_ObjectCategories",file))
    Len = len(Images)
    
    Pointer = int(0.8*Len)
    Image_names = []
    for j in range(Pointer,Len+1):
        Image_name = "{:03d}_{:04d}.jpg".format((i+1),j)
        os.system("rm {0}".format(os.path.join("data/Caltech/Train/256_ObjectCategories",file,Image_name)))

for i,file in enumerate(Test_Files):

    Images = os.listdir(os.path.join("data/Caltech/Test/256_ObjectCategories",file))
    Len = len(Images)
    
    Pointer = int(0.8*Len)
    Image_names = []
    for j in range(1,Pointer):
        Image_name = "{:03d}_{:04d}.jpg".format((i+1),j)
        os.system("rm {0}".format(os.path.join("data/Caltech/Test/256_ObjectCategories",file,Image_name)))
