import os
import numpy as np
import pandas as pd
import shutil
from utils.settings import param


class CSV:
    def __init__(self, tb_name: str, verbose: bool, z_ver: bool):
        self.verbose = verbose
        self.__set_current_table(tb_name)
        self.save_name = f"{param['data_dir']}/{self.tb_name}"
        self.save_sub = 'csv'
        self.save_path = f"{self.save_name}.{self.save_sub}"

        self.z_ver = z_ver
        if z_ver:
            name = self.save_name + '_z'
            self._set_name(name)
            if self.columns[-1] != param['z_cols'][-1]:
                self.columns += param['z_cols']

    def _set_name(self, name: str):
        self.save_name = name
        self.save_path = f"{self.save_name}.{self.save_sub}"

    def __set_current_table(self, tb_name):
        assert tb_name in param['tables']
        self.tb_name, self.columns = param[f"{tb_name}_table"], param[f"{tb_name}_cols"]

    def __drop(self):
        os.remove(self.save_path)

    def drop_if_exists(self):
        if self.csv_exists():
            print(f"Drop table from {self.save_path}")
            self.__drop()

    def csv_exists(self):
        return os.path.exists(self.save_path)


class Writer(CSV):
    def __init__(self, tb_name: str, verbose: bool, write_period: int, denormalize: bool, mean: pd.Series = None, stddev: pd.Series = None, z_ver: bool = False):
        super().__init__(tb_name, verbose, z_ver)
        self.write_period = write_period
        self.poses = None
        self.poses_cnt = 0
        self.write_cnt = 0
        self.save_path_old = f"{self.save_name}_o.{self.save_sub}"

        if write_period < float('inf'):
            # assert if denormalize then need to provide mean & stddev
            assert not (denormalize ^ (mean is not None and stddev is not None))
            self.__denormalize = denormalize
            if denormalize:
                self.__mean = mean.copy(deep=True)
                self.__stddev = stddev.copy(deep=True)

            try:
                shutil.copyfile(src=self.save_path, dst=self.save_path_old)
                print(f"Copy {self.save_path} to {self.save_path_old}")
            except Exception as e:
                print(f"Not exists an original data at {self.save_path}")
                    
            self.drop_if_exists()
    
    def cat_old(self):
        try:
            df = pd.read_csv(self.save_path_old)
            df.to_csv(self.save_path, mode='a', header=False, index=False)
            print(f"cat_old successfully from {self.save_path_old} to {self.save_path}")
        except Exception as e:
            print(f"No exists old data at {self.save_path_old}")
        

    def save(self):
        # TODO
        '''
        _summary_
        '''
        if self.poses is not None:
            # save poses if write_period
            df = pd.DataFrame(
                data=self.poses, columns=self.columns, dtype=np.float32)
            if self.__denormalize:
                if len(self.__mean) != len(self.columns):
                    len_diff = len(self.columns) - len(self.__mean)
                    self.__mean = np.concatenate(
                        (self.__mean, np.zeros((len_diff))))
                    self.__stddev = np.concatenate(
                        (self.__stddev, np.ones((len_diff))))
                df = df * self.__stddev + self.__mean

            if self.csv_exists():
                df.to_csv(self.save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.save_path, index=False)

            if self.verbose:
                print(
                    f"save {len(self.poses)}/{self.poses_cnt} to {self.save_path}")

            self.write_cnt += len(self.poses)
            self.poses = None

    def save_df(self, df, mode='w'):
        if mode == 'w':
            df.to_csv(self.save_path, index=False)
        else:
            df.to_csv(self.save_path, index=False, header=False, mode='a')
        if self.verbose:
            print(f"save {len(df)}/{len(df)} to {self.save_path}")

    def add_batch(self, batch_info):
        # TODO
        '''
        _summary_

        Parameters
        ----------
        batch_info : _type_
            _description_
        '''
        # append row_info into poses
        if self.poses is None:
            self.poses = np.array(batch_info)
        else:
            self.poses = np.row_stack((self.poses, batch_info))
        # update poses count
        self.poses_cnt += len(batch_info)
        # save poses
        if self.poses is not None and \
                len(self.poses) > self.write_period - 1:
            self.save()

        return self.write_cnt, self.poses_cnt

    def add_poses(self, row_info):
        # TODO
        '''
        _summary_

        Parameters
        ----------
        row_info : _type_
            _description_
        '''
        # append row_info into poses
        if self.poses is None:
            self.poses = np.array([row_info])
        else:
            self.poses = np.row_stack((self.poses, row_info))
        # update poses count
        self.poses_cnt += 1
        # save poses
        if self.poses is not None and \
                len(self.poses) > self.write_period - 1:
            self.save()
        return self.write_cnt, self.poses_cnt


class Reader(CSV):
    def __init__(self, tb_name: str, verbose: bool, enable_normalize: bool, z_ver: bool,
                 sort_by=[]):
        super().__init__(tb_name, verbose, z_ver)
        self.enable_normalize = enable_normalize
        self.data = None
        self.sort_by = sort_by

    def read_data(self):
        df = pd.read_csv(self.save_path)
        find_ml = df.columns == 'ml'
        if find_ml.any():
            df = df.astype({'ml': np.int8})

        if self.enable_normalize:
            self.true_mean = df.mean(numeric_only=True, axis=0)
            self.true_stddev = df.std(numeric_only=True, axis=0) + 1e-10
            df = (df - self.true_mean)/self.true_stddev
            print(f"Load normalized data from {self.save_path}")

        if len(self.sort_by) > 0:
            print(f"Load sorted data from {self.save_path}")
            df = df.sort_values(by=self.sort_by)

        self.data = df.to_numpy(dtype=np.float32)
        self.df = df
