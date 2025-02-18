

import config

from lib.util.data import read_preprocessed_file, load_json


user_ids = load_json("/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_user_ids.json")
users_history = read_preprocessed_file(
    "/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_train.txt"
) + read_preprocessed_file(
    "/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_valid.txt"
) + read_preprocessed_file(
    "/Users/dream/myProjects/dlkt-release/lab/settings/kg4ex_setting/statics2011_test.txt"
)