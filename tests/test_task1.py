import unittest
import json
import sys
import numpy as np
import os
import pandas as pd

if os.path.split(os.getcwd())[-1]=="Student_Local_Testing":
    folder_loc="main"
    sys.path.append(os.getcwd())
    from src.task1 import *
    from tests.utils import *
elif os.path.split(os.getcwd())[-1]=="tests":
    folder_loc="tests"
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
    from src.task1 import *
    from tests.utils import *
else:
    raise Exception(f"Running Tests from `{os.path.split(os.getcwd())[-1]}`. Please run the tests with your CWD set to either Student_Local_Testing or tests folders")

run_version_checks()

class Test_find_dataset_statistics(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"task1","sample.csv"), index_col=0)
        else:
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"..","task1","sample.csv"), index_col=0)
        self.target_col = "flagged"
        #Run Function Code
        self.n_records,self.n_columns,self.n_negative,self.n_positive,self.perc_positive = find_dataset_statistics(self.dataset,self.target_col)
    
    def test_nrecords(self):
        self.assertTrue(compare_submission_to_answer(self.n_records,10,"n_records"))

    def test_n_columns(self):
        self.assertTrue(compare_submission_to_answer(self.n_columns,6,"n_columns"))

    def test_n_negative(self):
        self.assertTrue(compare_submission_to_answer(self.n_negative,5,"n_negative"))

    def test_n_positive(self):
        self.assertTrue(compare_submission_to_answer(self.n_positive,5,"n_positive"))
    
    def test_perc_positive(self):
        self.assertTrue(compare_submission_to_answer(self.perc_positive,50,"perc_positive"))

class Test_find_data_type(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"task1","sample.csv"), index_col=0)
        else:
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"..","task1","sample.csv"), index_col=0)
        self.dataset["time"] = pd.to_datetime(self.dataset["time"])
    def test_find_data_type_src_ip(self):
        self.assertEqual(find_data_type(self.dataset, "src_ip"), object)

    def test_find_data_type_protocol(self):
        self.assertEqual(find_data_type(self.dataset, "protocol"), object)

    def test_find_data_type_bytes_in(self):
        self.assertEqual(find_data_type(self.dataset, "bytes_in"), np.int64)

    def test_find_data_type_bytes_out(self):
        self.assertEqual(find_data_type(self.dataset, "bytes_out"), np.int64)

    def test_find_data_type_time(self):
        self.assertEqual(find_data_type(self.dataset, "time"), np.dtype('datetime64[ns]'))

    def test_find_data_type_flagged(self):
        self.assertEqual(find_data_type(self.dataset, "flagged"), np.int64)

class Test_set_index_col(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"), index_col=0)
        pkl_files_folder = os.path.join(folder_path,"pkl_files")
        self.set_index_col_src_ip_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_src_ip.pkl"))
        self.set_index_col_protocol_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_protocol.pkl"))
        self.set_index_col_bytes_in_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_bytes_in.pkl"))
        self.set_index_col_bytes_out_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_bytes_out.pkl"))
        self.set_index_col_time_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_time.pkl"))
        self.set_index_col_flagged_ans = pd.read_pickle(os.path.join(pkl_files_folder,"set_index_col_flagged.pkl"))

    def test_set_index_col_src_ip(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["src_ip"]), self.set_index_col_src_ip_ans,"set_index_col (src_ip)"))

    def test_set_index_col_protocol(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["protocol"]), self.set_index_col_protocol_ans,"set_index_col (protocol)"))

    def test_set_index_col_bytes_in(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["bytes_in"]), self.set_index_col_bytes_in_ans,"set_index_col (bytes_in)"))

    def test_set_index_col_bytes_out(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["bytes_out"]), self.set_index_col_bytes_out_ans,"set_index_col (bytes_out)"))

    def test_set_index_col_time(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["time"]), self.set_index_col_time_ans,"set_index_col (time)"))
    
    def test_set_index_col_flagged(self):
        self.assertTrue(compare_submission_to_answer_df(set_index_col(self.dataset, self.dataset["flagged"]), self.set_index_col_flagged_ans,"set_index_col (flagged)"))
    
class Test_reset_index_col(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.dataset = pd.read_csv(os.path.join(folder_path,"reset_index_input.csv"), index_col=0)
        self.answer = pd.read_pickle(os.path.join(folder_path,"pkl_files","reset_index_col.pkl"))

    def test_reset_index_col(self):
        self.assertTrue(compare_submission_to_answer_df(reset_index_col(self.dataset),self.answer,"reset_index_col"))

class Test_set_col_type(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"task1","sample.csv"), index_col=0)
        else:
            self.dataset = pd.read_csv(os.path.join(os.getcwd(),"..","task1","sample.csv"), index_col=0)
        self.target_col = "flagged"

    def test_set_col_type_flagged(self):
        self.assertEqual(np.dtype(set_col_type(self.dataset, self.target_col,np.float64)[self.target_col]), np.float64)

class Test_sort_df_by_column(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"), index_col=0)
        self.sort_df_by_column_bytes_in_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","sort_bytes_in_desc.pkl"))
        self.sort_df_by_column_src_ip_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","sort_src_ip_asc.pkl"))

    def test_sort_df_bytes_in(self):
        self.assertTrue(compare_submission_to_answer_df(sort_DF_by_column(self.dataset, "bytes_in", True),self.sort_df_by_column_bytes_in_ans,"sort_DF_by_column (bytes_in)"))

    def test_sort_df_src_ip(self):
        self.assertTrue(compare_submission_to_answer_df(sort_DF_by_column(self.dataset, "src_ip", False),self.sort_df_by_column_src_ip_ans,"sort_DF_by_column (src_ip)"))

class Test_drop_na_cols(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")

        self.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"), index_col=0)
        self.drop_na_cols_ans =pd.read_pickle(os.path.join(folder_path,"pkl_files","drop_NA_cols.pkl"))
        
    def test_drop_na_cols(self):
        self.assertTrue(compare_submission_to_answer_df(drop_NA_cols(self.dataset2),self.drop_na_cols_ans,"drop_NA_cols"))

class Test_drop_na_rows(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")

        self.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"), index_col=0)
        self.drop_na_rows_ans =pd.read_pickle(os.path.join(folder_path,"pkl_files","drop_NA_rows.pkl"))

    def test_drop_na_rows(self):
        self.assertTrue(compare_submission_to_answer_df(drop_NA_rows(self.dataset2),self.drop_na_rows_ans,"drop_NA_rows"))

class Test_make_new_column(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"), index_col=0)
        self.make_new_column_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","make_new_column.pkl"))
        self.new_column_name = "New Column"
        self.new_column_values = [12,13,14,15,16,17,18,19,20,21]
        
    def test_make_new_column(self):
        self.assertTrue(compare_submission_to_answer_df(make_new_column(self.dataset, self.new_column_name, self.new_column_values),self.make_new_column_ans,"make_new_column"))

class Test_left_merge_DFs_by_column(unittest.TestCase):
    def setUp(self):
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.dataset = pd.read_csv(os.path.join(folder_path,"sample.csv"), index_col=0)
        self.dataset2 = pd.read_csv(os.path.join(folder_path,"sample2.csv"), index_col=0)
        self.left_merge_dfs_by_column_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","left_merge_DFs_by_column.pkl"))
        self.merge_col = "src_ip"

    def test_left_merge_DFs_by_column(self):
        self.assertTrue(compare_submission_to_answer_df(left_merge_DFs_by_column(self.dataset, self.dataset2, self.merge_col),self.left_merge_dfs_by_column_ans,"left_merge_DFs_by_column"))

class Test_make_df_from_2d_array(unittest.TestCase):
    def setUp(self):
        self.array_2d = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23],[24,25,26],[27,28,29]])
        self.column_names = ["A","B","C"]
        self.index = pd.Series([0,5,10,15,20,25,30,35,40,45])
        if folder_loc=="main":
            folder_path = os.path.join(os.getcwd(),"task1")
        else:
            folder_path = os.path.join(os.getcwd(),"..","task1")
        self.make_df_from_2d_array_ans = pd.read_pickle(os.path.join(folder_path,"pkl_files","make_DF_from_2d_array.pkl"))
    def test_make_df_from_2d_array(self):
        self.assertTrue(compare_submission_to_answer_df(make_DF_from_2d_array(self.array_2d,self.column_names,self.index),self.make_df_from_2d_array_ans,"make_DF_from_2d_array"))

class Test_simpleClass(unittest.TestCase):
    def setUp(self):
        self.simpleClass = simpleClass(length=1, width=2, height=3)

    def test_simpleClass_length(self):
        self.assertEqual(self.simpleClass.length, 1)
    
    def test_simpleClass_width(self):
        self.assertEqual(self.simpleClass.width, 2)

    def test_simpleClass_height(self):
        self.assertEqual(self.simpleClass.height, 3)

if __name__ == '__main__':
    unittest.main()
