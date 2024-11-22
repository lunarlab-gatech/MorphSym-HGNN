from .flexibleDataset import FlexibleDataset
import os
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

class Solo12Dataset(FlexibleDataset):
    """
    Dataset class for the Solo12 robot data.
    Each leg has 3 joints: HAA (Hip Abduction/Adduction),
    HFE (Hip Flexion/Extension), and KFE (Knee Flexion/Extension)
    """
    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_name(self):
        """Specify the name of the raw data file"""
        return "data.npz"
    
    def get_file_id_and_loc(self):
        """Specify the source location of the data file"""
        # If loading data from local, return local path
        return "local_file", "local"
        # If data needs to be downloaded, return download link
        # return "https://path/to/your/data_100000.npz", "url"
    
    def get_expected_urdf_name(self):
        return "solo"
    
    # ======================== DATA LOADING ==========================
    def get_urdf_name_to_dataset_array_index(self):
        """Joint mapping for Solo12"""
        '''
            # The data structure in npz file should be:
            X = [
                # Joint positions (first 12 columns)
                'FL_HAA', 'FL_HFE', 'FL_KFE',  # Front Left leg (0-2)
                'FR_HAA', 'FR_HFE', 'FR_KFE',  # Front Right leg (3-5)
                'HL_HAA', 'HL_HFE', 'HL_KFE',  # Hind Left leg (6-8)
                'HR_HAA', 'HR_HFE', 'HR_KFE',  # Hind Right leg (9-11)
                
                # Joint velocities (last 12 columns)
                'FL_HAA_vel', 'FL_HFE_vel', 'FL_KFE_vel',
                'FR_HAA_vel', 'FR_HFE_vel', 'FR_KFE_vel',
                'HL_HAA_vel', 'HL_HFE_vel', 'HL_KFE_vel',
                'HR_HAA_vel', 'HR_HFE_vel', 'HR_KFE_vel'
            ]

            Y = [
                'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',  # Base linear velocity (0-2)
                'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z'   # Base angular velocity (3-5)
            ]
        '''
        return {
            'FL_HAA': 0,   # Front Left Hip Abduction/Adduction
            'FL_HFE': 1,   # Front Left Hip Flexion/Extension
            'FL_KFE': 2,   # Front Left Knee Flexion/Extension
            'FR_HAA': 3,   # Front Right Hip
            'FR_HFE': 4,   # Front Right Hip
            'FR_KFE': 5,   # Front Right Knee
            'HL_HAA': 6,   # Hind Left Hip
            'HL_HFE': 7,   # Hind Left Hip
            'HL_KFE': 8,   # Hind Left Knee
            'HR_HAA': 9,   # Hind Right Hip
            'HR_HFE': 10,  # Hind Right Hip
            'HR_KFE': 11   # Hind Right Knee
        }

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        """Load data for a specified sequence"""
        # Load joint data
        j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        
        # Load base velocity data (as labels)
        base_lin_vel = np.array(self.mat_data['base_lin_vel'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        base_ang_vel = np.array(self.mat_data['base_ang_vel'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        
        # Concatenate base velocities into a 6D vector as labels
        labels = np.concatenate([base_lin_vel[-1], base_ang_vel[-1]])  # Only use the last frame of the sequence as prediction target
        
        # Fields returned as None are used in original QuadSDKDataset but not needed here
        return None, None, j_p, j_v, None, None, None, labels, None, None, None

    # ======================== DATA PROCESSING =======================
    def process(self):
        """
        Process raw Solo-12 data and save it in .mat format
        Data format:
        - X: (N, 24) contains positions and velocities of 12 joints
        - Y: (N, 6) contains base linear and angular velocities
        """
        # Read the npz file
        path_to_npz = os.path.join(self.root, 'raw', 'data.npz')
        raw_data = np.load(path_to_npz)
        
        # Get the data length
        dataset_entries = len(raw_data['X'])
        
        # Separate the joint positions and velocities
        joint_positions = raw_data['X'][:, :12]    # The first 12 columns are joint positions
        joint_velocities = raw_data['X'][:, 12:]   # The last 12 columns are joint velocities
        
        # Separate the base linear and angular velocities
        base_lin_vel = raw_data['Y'][:, :3]        # The first 3 columns are linear velocities
        base_ang_vel = raw_data['Y'][:, 3:]        # The last 3 columns are angular velocities
        
        # Create the data dictionary
        data_dict = {
            'q': joint_positions,           # Joint positions (N, 12)
            'qd': joint_velocities,         # Joint velocities (N, 12)
            'base_lin_vel': base_lin_vel,   # Base linear velocities (N, 3)
            'base_ang_vel': base_ang_vel,   # Base angular velocities (N, 3)
            'timestamps': np.arange(dataset_entries),  # Use the index as timestamps
        }
        
        # Save as a mat file
        sio.savemat(os.path.join(self.processed_dir, "data.mat"), 
                    data_dict, 
                    do_compression=True)
        
        # Save the dataset information
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            file_id, loc = self.get_file_id_and_loc()
            f.write(str(dataset_entries) + " " + file_id)

    def swap_legs_data(self, data_dict, leg1_idx, leg2_idx):
        """
        Swap the data of two legs
        leg_idx mapping: FR=0, FL=1, HR=2, HL=3
        Each leg has 3 joints
        """
        # Calculate the index range of the joints
        leg1_start = leg1_idx * 3
        leg1_end = leg1_start + 3
        leg2_start = leg2_idx * 3
        leg2_end = leg2_start + 3
        
        # Swap the joint position data
        if 'j_p' in data_dict and data_dict['j_p'] is not None:
            data_dict['j_p'][:, leg1_start:leg1_end], data_dict['j_p'][:, leg2_start:leg2_end] = \
                data_dict['j_p'][:, leg2_start:leg2_end].copy(), data_dict['j_p'][:, leg1_start:leg1_end].copy()
        
        # Swap the joint velocity data
        if 'j_v' in data_dict and data_dict['j_v'] is not None:
            data_dict['j_v'][:, leg1_start:leg1_end], data_dict['j_v'][:, leg2_start:leg2_end] = \
                data_dict['j_v'][:, leg2_start:leg2_end].copy(), data_dict['j_v'][:, leg1_start:leg1_end].copy()
        
        # Swap the joint torque data
        if 'j_T' in data_dict and data_dict['j_T'] is not None:
            data_dict['j_T'][:, leg1_start:leg1_end], data_dict['j_T'][:, leg2_start:leg2_end] = \
                data_dict['j_T'][:, leg2_start:leg2_end].copy(), data_dict['j_T'][:, leg1_start:leg1_end].copy()
        
        return data_dict