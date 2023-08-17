from config import *

config = configuration()
random_seed(config.seed)


class extractPath:
    # make dataFrame of pandas in order to list datasets

    def __init__(self) -> None:
        print("Extracting dataset path done !!\n")
        self.survival_info = pd.read_csv(config.train_path + '/survival_info.csv')
        self.name_mapping = pd.read_csv(config.train_path + '/name_mapping.csv')
        self.name_mapping.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) # rename to the patients dataSet
        df = self.survival_info.merge(self.name_mapping, on="Brats20ID", how="right") # add mapping names into survival info
        # print(df.head(10))

        self.data_paths = list()
        for index, row  in df.iterrows():
            patient_id = row['Brats20ID']
            phase = patient_id.split("_")[-2]
            
            if phase == 'Training':
                path = os.path.join(config.train_path, patient_id)
            else:
                path = os.path.join(config.test_path, patient_id)
            self.data_paths.append(path)
            
        df['path'] = self.data_paths

        self.train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
        self.train_data = self.train_data.loc[self.train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, ) # remove patinet 355 because it's results are false
         # save data train structure

    def train_test_valid(self):
        print("Spliting data path into train test valid done !!\n")
        # Kfold = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True) 
        # for i, (train_index, val_index) in enumerate(Kfold.split(self.train_data, self.train_data["Age"]//10*10)):
        #         self.train_data.loc[val_index, "fold"] = i

        # train_df = self.train_data.loc[self.train_data['fold'] != 0].reset_index(drop=True)
        # val_df = self.train_data.loc[self.train_data['fold'] == 0].reset_index(drop=True)
        # test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
        
        Kfold = StratifiedKFold(n_splits=10, random_state=config.seed, shuffle=True)
        for i, (train_index, val_index) in enumerate(Kfold.split(self.train_data, self.train_data["Age"] // 10 * 10)):
            self.train_data.loc[val_index, "fold"] = i

        # Extract 10% test data
        test_indices = self.train_data[self.train_data['fold'] == 9].index
        test_df = self.train_data.loc[test_indices].reset_index(drop=True)

        # Extract 10% validation data
        val_indices = self.train_data[self.train_data['fold'] == 0].index
        val_df = self.train_data.loc[val_indices].reset_index(drop=True)

        # Extract 80% training data
        train_indices = self.train_data[(self.train_data['fold'] != 9) & (self.train_data['fold'] != 0)].index
        train_df = self.train_data.loc[train_indices].reset_index(drop=True)    
        self.train_data.to_csv(config.train_csv_path, index=False)

        return train_df, val_df, test_df
    
    def plotResult(self, train_df, val_df, test_df):
        dataframes = ['train_df', 'val_df', 'test_df']
        shapes = [train_df.shape[0], val_df.shape[0], test_df.shape[0]]
        plt.bar(dataframes, shapes, color=['blue', 'green', 'orange'])
        plt.xlabel('DataFrames')
        plt.ylabel('Number of Rows')
        plt.title('Number of Rows in DataFrames')
        plt.show()


class generator(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test"):
        # print("Generating and Set augmentation on data done !!\n")
        self.df = df # calea
        self.phase = phase
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii'] 
        self.augmentations = self.get_augmentations(phase)
        
        
    def __len__(self):
        return self.df.shape[0] 
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']  
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0] 
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            


            self.get_center_crop_coords(240,240,155, 128,128,128)
            img=self.center_crop(img, 128,128,128)
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1)) 
        

        
        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii") 
            mask = self.load_img(mask_path)

            
            self.get_center_crop_coords(240,240,155, 128,128,128)
            mask=self.center_crop(mask, 128,128,128)
            mask = self.preprocess_mask_labels(mask)
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        
        return {
            "Id": id_,
            "image": img,
        }

    def get_augmentations(self, phase):
        list_transforms = []
        list_trfms = Compose(list_transforms)
        return list_trfms


    def get_augmentation_v1(self, patch_size):
        return Compose([
            Rotate((-30, -15, 15, 30), (0, 0), (0, 0), p=0.5),
            Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
        ], p=1.0)
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)



    def get_center_crop_coords(self,height, width, depth, crop_height, crop_width, crop_depth):
                x1 = (height - crop_height) // 2
                x2 = x1 + crop_height
                y1 = (width - crop_width) // 2
                y2 = y1 + crop_width
                z1 = (depth - crop_depth) // 2
                z2 = z1 + crop_depth
                return x1, y1, z1, x2, y2, z2

    def center_crop(self, data:np.ndarray, crop_height, crop_width, crop_depth):
        height, width, depth = data.shape[:3]
        if height < crop_height or width < crop_width or depth < crop_depth:
            raise ValueError
        x1, y1, z1, x2, y2, z2 = self.get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
        data = data[x1:x2, y1:y2, z1:z2]
        return data

    
    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1 # eticheta 1 = necrotic / non-enhancing tumor core
        mask_WT[mask_WT == 2] = 1 # eticheta 2 = peritumoral edema
        mask_WT[mask_WT == 4] = 1 # eticheta 4 = enhancing tumor core

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1)) # mutam axele pentru a putea vizualiza mastile ulterior

        return mask